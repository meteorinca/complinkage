"""
Real-World Linkage Topology Optimizer
=====================================
A user-friendly wrapper around the research-grade linkage topology optimizer
for practical 3D-printed mechanisms.

Quick-start:
    >>> import real_world_linkage_runner as rwl
    >>> design = rwl.quick_design(
    ...     motion_data=your_motion,
    ...     ground_a=(0.0, 0.0),
    ...     ground_b=(80.0, 0.0),
    ...     motor_axis=(80.0, 0.0),
    ...     output_dir="./my_linkage_output",
    ... )

Features:
    - Import motion from arrays, CSV files, or JSON
    - Automatic result caching: re-running the same data skips optimization
    - Real-world mm unit support for 3D printing
    - Multi-link collision layering
    - Full diagnostic plots

For help with input data formats, see easywayinput.md
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Any
import hashlib
import json
import math
import pickle

import numpy as np
from numpy.typing import ArrayLike, NDArray

import linkage_topology_optimizer as lto


# ---------------------------------------------------------------------------
# Caching helpers
# ---------------------------------------------------------------------------

def _array_to_hash(arr: NDArray[np.float64]) -> str:
    """Stable hash of a float array (rounds to avoid float-noise mismatches)."""
    rounded = np.round(arr, 8)
    return hashlib.sha256(rounded.tobytes()).hexdigest()[:16]


def _compute_cache_key(
    motion: "MotionData",
    params: "DesignParameters",
) -> str:
    """Produce a short hash from motion data + key parameters.
    
    Only parameters that affect the optimization result are included.
    Fabrication-only params (thickness, width, hole_radius) are NOT
    included — so you can change those and re-export without re-optimizing.
    """
    # Motion data
    parts = [
        _array_to_hash(motion.states_a),
        _array_to_hash(motion.states_b),
        _array_to_hash(motion.motor_positions),
        _array_to_hash(motion.ground_a),
        _array_to_hash(motion.ground_b),
    ]
    # Optimization-affecting params (NOT fabrication)
    opt_params = {
        "n_samples": params.n_samples,
        "n_starts": params.n_starts,
        "n_candidates": params.n_candidates,
        "gamma": params.gamma,
        "area_mode": params.area_mode,
        "minimum_moment_arm_fraction": params.minimum_moment_arm_fraction,
        "cma_max_iterations": params.cma_max_iterations,
        "cma_population_size": params.cma_population_size,
        "cma_time_stride": params.cma_time_stride,
        "cma_sigma0": params.cma_sigma0,
        "bend_fraction": params.bend_fraction,
        "tangent_fraction": params.tangent_fraction,
        "solver_max_iterations": params.solver_max_iterations,
        "solver_tolerance": params.solver_tolerance,
    }
    parts.append(hashlib.sha256(json.dumps(opt_params, sort_keys=True).encode()).hexdigest()[:16])
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]


def _save_cache(cache_dir: Path, cache_key: str, result_data: Dict[str, Any]) -> None:
    """Save optimization results to disk."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Split: numpy arrays → .npz, scalars+lists → .json
    npz_data: Dict[str, NDArray[np.float64]] = {}
    json_data: Dict[str, Any] = {}
    
    for key, value in result_data.items():
        if isinstance(value, np.ndarray):
            npz_data[key] = value
        elif isinstance(value, (list, tuple)) and len(value) > 0 and isinstance(value[0], (int, float, np.floating)):
            npz_data[key] = np.asarray(value, dtype=float)
        else:
            json_data[key] = value
    
    np.savez_compressed(cache_dir / f"{cache_key}.npz", **npz_data)
    with open(cache_dir / f"{cache_key}.json", "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2, default=str)


def _load_cache(cache_dir: Path, cache_key: str) -> Optional[Dict[str, Any]]:
    """Load cached optimization results. Returns None if not found."""
    npz_path = cache_dir / f"{cache_key}.npz"
    json_path = cache_dir / f"{cache_key}.json"
    if not npz_path.exists() or not json_path.exists():
        return None
    
    data: Dict[str, Any] = {}
    npz = np.load(npz_path)
    for key in npz.files:
        data[key] = npz[key]
    npz.close()
    
    with open(json_path, "r", encoding="utf-8") as f:
        json_data = json.load(f)
    data.update(json_data)
    return data


def _reconstruct_candidate(cache: Dict[str, Any]) -> lto.TopologyCandidate:
    """Reconstruct a TopologyCandidate from cached data."""
    return lto.TopologyCandidate(
        q_a=np.asarray(cache["q_a"], dtype=float),
        q_b=np.asarray(cache["q_b"], dtype=float),
        length=float(cache["length"]),
        objective=float(cache["objective"]),
        distance_variance=float(cache["distance_variance"]),
        distance_cv=float(cache["distance_cv"]),
        minimum_moment_arm=float(cache["minimum_moment_arm"]),
        percentile05_moment_arm=float(cache["percentile05_moment_arm"]),
        mean_moment_arm=float(cache["mean_moment_arm"]),
        area_energy=float(cache["area_energy"]),
        converged=bool(cache["converged"]),
        iterations=int(cache.get("iterations", 0)),
        start=np.asarray(cache.get("start", [0, 0, 0, 0]), dtype=float),
        history=list(cache.get("history", [])),
    )


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class MotionData:
    """Container for two-body periodic motion input.

    Parameters
    ----------
    states_a : (n_frames, 3) array
        States of body A: [theta_rad, world_x, world_y] per frame.
    states_b : (n_frames, 3) array
        States of body B: [theta_rad, world_x, world_y] per frame.
    motor_positions : (n_frames, 2) array
        World position of the motor/input pivot per frame.
        For a crank, this is the crank ground pivot (often constant).
    ground_a : (2,) array
        Ground pivot of body A (constant).
    ground_b : (2,) array
        Ground pivot of body B (constant).
    times : (n_frames,) array or None
        Phase values. If None, linspace(0, 2π, n_frames) is used.
    label : str
        Human-readable name for this motion dataset.
    """
    states_a: NDArray[np.float64]
    states_b: NDArray[np.float64]
    motor_positions: NDArray[np.float64]
    ground_a: NDArray[np.float64]
    ground_b: NDArray[np.float64]
    times: Optional[NDArray[np.float64]] = None
    label: str = "custom_motion"

    def __post_init__(self):
        self.states_a = np.asarray(self.states_a, dtype=float)
        self.states_b = np.asarray(self.states_b, dtype=float)
        self.motor_positions = np.asarray(self.motor_positions, dtype=float)
        self.ground_a = np.asarray(self.ground_a, dtype=float).ravel()
        self.ground_b = np.asarray(self.ground_b, dtype=float).ravel()
        n = len(self.states_a)
        if self.times is None:
            self.times = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
        else:
            self.times = np.asarray(self.times, dtype=float)
        if not (len(self.states_b) == n == len(self.motor_positions)):
            raise ValueError(
                f"states_a ({len(self.states_a)}), states_b ({len(self.states_b)}), "
                f"motor_positions ({len(self.motor_positions)}) must have equal lengths."
            )
        if len(self.ground_a) != 2 or len(self.ground_b) != 2:
            raise ValueError("ground_a and ground_b must be 2-element arrays.")

    @property
    def n_frames(self) -> int:
        return len(self.states_a)

    @property
    def target_states(self) -> NDArray[np.float64]:
        return np.stack([self.states_a, self.states_b], axis=1)

    @property
    def characteristic_length(self) -> float:
        cloud = np.vstack([
            self.states_a[:, 1:3],
            self.states_b[:, 1:3],
            self.motor_positions,
        ])
        span = np.ptp(cloud, axis=0)
        return max(float(np.linalg.norm(span)), 1.0)


@dataclass
class BodySpec:
    """Geometry specification for a single rigid body.

    Parameters
    ----------
    name : str
    outline : (n_vertices, 2) array or None
        Polygon vertices in local coords. Default: unit rectangle.
    bone_segment : ((2,),(2,)) or None
        Bone as (start, end) in local coords. Default: outline x-extent.
    markers : dict[str, (2,)]
        Named marker points, e.g. {"tip": [25.0, 0.0]}.
    """
    name: str
    outline: Optional[NDArray[np.float64]] = None
    bone_segment: Optional[Tuple[NDArray[np.float64], NDArray[np.float64]]] = None
    markers: Dict[str, NDArray[np.float64]] = field(default_factory=dict)

    def __post_init__(self):
        if self.outline is None:
            self.outline = np.array([[-0.1, -0.1], [1.0, -0.1], [1.0, 0.1], [-0.1, 0.1]], dtype=float)
        else:
            self.outline = np.asarray(self.outline, dtype=float)
        if self.bone_segment is None:
            xs = self.outline[:, 0]
            self.bone_segment = (np.array([xs.min(), 0.0]), np.array([xs.max(), 0.0]))
        else:
            self.bone_segment = (
                np.asarray(self.bone_segment[0], dtype=float),
                np.asarray(self.bone_segment[1], dtype=float),
            )
        self.markers = {k: np.asarray(v, dtype=float) for k, v in self.markers.items()}

    def to_rigid_component(self) -> lto.RigidComponent:
        return lto.RigidComponent(
            name=self.name,
            outline=self.outline,
            bone_segment=self.bone_segment,
            markers=self.markers,
        )


@dataclass
class DesignParameters:
    """All tunable knobs.

    Parameters
    ----------
    n_starts : int
        Multi-start count (default 80). Higher = more thorough.
    n_candidates : int
        Distinct designs to return (default 8).
    gamma : float
        Moment-arm penalty weight (default 0.1).
    area_mode : str
        "barrier" (recommended) or "paper".
    cma_max_iterations : int
        CMA-ES generations (default 50). 80-200 for final runs.
    cma_time_stride : int
        Subsample for speed (default 3). Set 1 for final.
    fabrication_thickness : float
        Link thickness in mm (default 3.0).
    fabrication_width : float
        Link width in mm (default 5.0).
    hole_radius : float
        Pin hole radius in mm (default 2.0).
    unit_scale : float
        Abstract units per mm (default 1.0 = data is in mm).
    bend_fraction : float
        Link bend as fraction of length (0 = straight).
    cache_dir : str or Path
        Where to store cached optimization results.
    force_rerun : bool
        If True, skip cache and re-optimize.
    """
    # topology search
    n_samples: int = 121
    n_starts: int = 80
    n_candidates: int = 8
    gamma: float = 0.1
    coordinate_bound: Optional[float] = None
    area_mode: str = "barrier"
    minimum_moment_arm_fraction: float = 0.005
    seed: int = 7

    # CMA-ES
    cma_max_iterations: int = 50
    cma_population_size: Optional[int] = None
    cma_time_stride: int = 3
    cma_sigma0: float = 0.10

    # fabrication (not included in cache key)
    fabrication_thickness: float = 3.0
    fabrication_width: float = 5.0
    hole_radius: float = 2.0
    unit_scale: float = 1.0

    # organic shaping
    bend_fraction: float = 0.08
    tangent_fraction: float = 0.18

    # marker specs for CMA-ES
    marker_specs: Optional[List[Tuple[int, NDArray[np.float64]]]] = None

    # solver knobs
    solver_max_iterations: int = 40
    solver_tolerance: float = 1e-9

    # caching
    cache_dir: str = "./linkage_cache"
    force_rerun: bool = False


@dataclass
class DesignResult:
    """Output of a full linkage design run."""
    motion: MotionData
    params: DesignParameters
    components: List[lto.RigidComponent]
    candidates: List[lto.TopologyCandidate]
    best_candidate: lto.TopologyCandidate
    cma_result: Optional[lto.CMAResult]
    simulation: lto.SimulationResult
    system: lto.ConstraintSystem
    centerline: NDArray[np.float64]
    spline_metadata: Dict[str, Any]
    stl_path: Path
    json_path: Path
    from_cache: bool = False

    def summary(self) -> str:
        bc = self.best_candidate
        src = "CACHED" if self.from_cache else "OPTIMIZED"
        lines = [
            "=" * 58,
            f"  LINKAGE DESIGN RESULT  [{src}]",
            "=" * 58,
            f"  Motion:       {self.motion.label} ({self.motion.n_frames} frames)",
            f"  Pin A:        q_a = [{bc.q_a[0]:.3f}, {bc.q_a[1]:.3f}]",
            f"  Pin B:        q_b = [{bc.q_b[0]:.3f}, {bc.q_b[1]:.3f}]",
            f"  Link length:  {bc.length:.3f} abstract units",
            f"  Dist CV:      {bc.distance_cv:.2e}",
            f"  P05 mom.arm:  {bc.percentile05_moment_arm:.3f}",
            f"  Min mom.arm:  {bc.minimum_moment_arm:.3f}",
            f"  Converged:    {bc.converged}",
            "─" * 58,
            f"  Fab thickness:{self.params.fabrication_thickness:.1f} mm",
            f"  Fab width:    {self.params.fabrication_width:.1f} mm",
            f"  Hole radius:  {self.params.hole_radius:.2f} mm",
            f"  Unit scale:   {self.params.unit_scale:.3f} abs/mm",
            "─" * 58,
            f"  STL:          {self.stl_path}",
            f"  JSON:         {self.json_path}",
            "=" * 58,
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Motion creation helpers
# ---------------------------------------------------------------------------

def create_motion_from_arrays(
    states_a: ArrayLike,
    states_b: ArrayLike,
    ground_a: ArrayLike = (0.0, 0.0),
    ground_b: ArrayLike = (3.0, 0.0),
    motor_positions: Optional[ArrayLike] = None,
    times: Optional[ArrayLike] = None,
    label: str = "custom_motion",
) -> MotionData:
    """Create MotionData from raw state arrays.

    Parameters
    ----------
    states_a, states_b : (n_frames, 3) array_like
        Body states [theta_rad, world_x, world_y].
    ground_a, ground_b : (2,) array_like
    motor_positions : (n_frames, 2) array_like or None
        Motor world positions. None = use ground_a repeated.
    times : (n_frames,) array_like or None
    label : str
    """
    sa = np.asarray(states_a, dtype=float)
    sb = np.asarray(states_b, dtype=float)
    n = len(sa)
    if motor_positions is None:
        mp = np.repeat(np.asarray(ground_a, dtype=float)[None, :], n, axis=0)
    else:
        mp = np.asarray(motor_positions, dtype=float)
    return MotionData(
        states_a=sa, states_b=sb,
        motor_positions=mp,
        ground_a=np.asarray(ground_a, dtype=float),
        ground_b=np.asarray(ground_b, dtype=float),
        times=None if times is None else np.asarray(times, dtype=float),
        label=label,
    )


def create_crank_rocker_motion(
    ground_a: ArrayLike = (0.0, 0.0),
    ground_b: ArrayLike = (80.0, 0.0),
    crank_radius: float = 25.0,
    rocker_radius: float = 55.0,
    coupler_length: float = 62.5,
    n_samples: int = 121,
    phase_offset: float = 0.0,
    initial_branch: str = "upper",
) -> MotionData:
    """Generate motion for a classical crank-rocker four-bar mechanism.

    Useful for testing. The optimizer will try to rediscover the hidden coupler.
    All units should match whatever unit system you're using (e.g. mm).
    """
    times = np.linspace(0.0, 2.0 * np.pi, n_samples, endpoint=False)
    oa = np.asarray(ground_a, dtype=float)
    ob = np.asarray(ground_b, dtype=float)
    states = np.zeros((n_samples, 2, 3), dtype=float)
    previous = None

    for i, t in enumerate(times):
        theta_a = float(t + phase_offset)
        pa = oa + lto.rotation(theta_a) @ np.array([crank_radius, 0.0])
        delta = ob - pa
        d = float(np.linalg.norm(delta))
        if d < 1e-12 or d > coupler_length + rocker_radius + 1e-10 or d < abs(coupler_length - rocker_radius) - 1e-10:
            raise ValueError(f"Crank-rocker dimensions do not close at phase {t:.3f}.")
        a = (coupler_length**2 - rocker_radius**2 + d**2) / (2.0 * d)
        h2 = max(coupler_length**2 - a**2, 0.0)
        h = math.sqrt(h2)
        unit = delta / d
        base = pa + a * unit
        perp = np.array([-unit[1], unit[0]])
        p_plus = base + h * perp
        p_minus = base - h * perp
        if previous is None:
            chosen = p_plus if initial_branch.lower() == "upper" else p_minus
        else:
            chosen = p_plus if np.linalg.norm(p_plus - previous) <= np.linalg.norm(p_minus - previous) else p_minus
        previous = chosen
        theta_b = math.atan2(chosen[1] - ob[1], chosen[0] - ob[0])
        states[i, 0] = [theta_a, oa[0], oa[1]]
        states[i, 1] = [theta_b, ob[0], ob[1]]

    motor_positions = np.repeat(oa[None, :], n_samples, axis=0)
    return MotionData(
        states_a=states[:, 0], states_b=states[:, 1],
        motor_positions=motor_positions,
        ground_a=oa, ground_b=ob, times=times,
        label=f"crank_rocker_{crank_radius}_{rocker_radius}_{coupler_length}",
    )


def load_motion_from_csv(
    path_a: str | Path,
    path_b: str | Path,
    ground_a: ArrayLike,
    ground_b: ArrayLike,
    motor_positions_path: Optional[str | Path] = None,
    delimiter: str = ",",
    skip_header: int = 0,
    columns: Tuple[int, int, int] = (0, 1, 2),
    label: str = "csv_motion",
) -> MotionData:
    """Load body motion from CSV files.

    Each CSV should have columns [theta, x, y] (reorderable via `columns`).
    Both files must have the same number of rows.

    Parameters
    ----------
    path_a, path_b : str or Path
    ground_a, ground_b : (2,) array_like
    motor_positions_path : str, Path, or None
        CSV for motor positions (2 cols: x, y). None = use ground_a.
    delimiter : str
    skip_header : int
    columns : (col_theta, col_x, col_y)
    label : str
    """
    def _read(path, cols):
        raw = np.loadtxt(path, delimiter=delimiter, skiprows=skip_header)
        return np.column_stack([raw[:, c] for c in cols])

    sa = _read(path_a, columns)
    sb = _read(path_b, columns)
    if motor_positions_path is not None:
        mp = np.loadtxt(motor_positions_path, delimiter=delimiter, skiprows=skip_header)
        if mp.ndim == 1:
            mp = mp.reshape(-1, 2)
    else:
        mp = np.repeat(np.asarray(ground_a, dtype=float)[None, :], len(sa), axis=0)
    return create_motion_from_arrays(sa, sb, ground_a, ground_b, mp, label=label)


def load_motion_from_json(path: str | Path, label: str = "json_motion") -> MotionData:
    """Load motion from a JSON file.

    Expected format:
    {
        "states_a": [[theta, x, y], ...],
        "states_b": [[theta, x, y], ...],
        "ground_a": [x, y],
        "ground_b": [x, y],
        "motor_positions": [[x, y], ...],   // optional
        "times": [t0, t1, ...]              // optional
    }
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return create_motion_from_arrays(
        states_a=data["states_a"],
        states_b=data["states_b"],
        ground_a=data.get("ground_a", [0.0, 0.0]),
        ground_b=data.get("ground_b", [3.0, 0.0]),
        motor_positions=data.get("motor_positions"),
        times=data.get("times"),
        label=label,
    )


# ---------------------------------------------------------------------------
# DXF import helper
# ---------------------------------------------------------------------------

def load_motion_from_dxf_frames(
    frame_files_a: List[str | Path],
    frame_files_b: List[str | Path],
    ground_a: ArrayLike,
    ground_b: ArrayLike,
    layer_name: str = "0",
    entity_handles: Optional[Tuple[str, str]] = None,
    motor_positions: Optional[ArrayLike] = None,
    label: str = "dxf_motion",
) -> MotionData:
    """Load motion from a sequence of DXF frame exports.

    Use this when you've exported each frame of your CAD animation as a
    separate DXF file. Each DXF should contain the 2D geometry of one body
    at one time step.  The function extracts the position and orientation
    of a specified LINE entity in each frame.

    Requires: ezdxf  (`pip install ezdxf`)

    Parameters
    ----------
    frame_files_a : list of str/Path
        DXF files for body A, one per frame, in order.
    frame_files_b : list of str/Path
        DXF files for body B, one per frame, in order.
    ground_a, ground_b : (2,) array_like
        Ground pivot positions.
    layer_name : str
        DXF layer to look for the tracking entity on.
    entity_handles : (str, str) or None
        DXF handles of the specific LINE entities to track for body A and
        body B. If None, the first LINE on the specified layer is used.
    motor_positions : (n_frames, 2) array_like or None
    label : str

    Returns
    -------
    MotionData

    Notes
    -----
    The function extracts [theta, x, y] from each frame by:
      - Taking the LINE entity's midpoint as (x, y)
      - Computing theta from the LINE's direction vector
    For bodies that rotate about a fixed pivot, the pivot itself should be
    used as ground_a/ground_b; the LINE's midpoint should be a point on the
    body away from the pivot (e.g. the distal end).
    """
    try:
        import ezdxf
    except ImportError:
        raise ImportError(
            "DXF import requires ezdxf. Install with: pip install ezdxf"
        )

    def _extract_state(filepath, handle_hint):
        doc = ezdxf.readfile(str(filepath))
        msp = doc.modelspace()
        # Find lines on the specified layer
        candidates = [
            e for e in msp.query("LINE")
            if e.dxf.layer == layer_name
        ]
        if not candidates:
            raise ValueError(f"No LINE entities found on layer '{layer_name}' in {filepath}")
        
        # If a specific handle is requested, try to find it
        line = None
        if handle_hint is not None:
            for c in candidates:
                if c.dxf.handle == handle_hint:
                    line = c
                    break
        if line is None:
            line = candidates[0]
        
        start = np.array([line.dxf.start.x, line.dxf.start.y], dtype=float)
        end = np.array([line.dxf.end.x, line.dxf.end.y], dtype=float)
        midpoint = 0.5 * (start + end)
        direction = end - start
        theta = math.atan2(direction[1], direction[0])
        return np.array([theta, midpoint[0], midpoint[1]], dtype=float)

    n = len(frame_files_a)
    if len(frame_files_b) != n:
        raise ValueError(
            f"frame_files_a ({len(frame_files_a)}) and frame_files_b "
            f"({len(frame_files_b)}) must have equal lengths."
        )

    handle_a, handle_b = entity_handles if entity_handles else (None, None)
    states_a = np.array([_extract_state(f, handle_a) for f in frame_files_a])
    states_b = np.array([_extract_state(f, handle_b) for f in frame_files_b])

    return create_motion_from_arrays(
        states_a, states_b, ground_a, ground_b,
        motor_positions=motor_positions, label=label,
    )


# ---------------------------------------------------------------------------
# Pipeline (with caching)
# ---------------------------------------------------------------------------

def _auto_marker_specs(
    components: List[lto.RigidComponent],
    params: DesignParameters,
) -> List[Tuple[int, NDArray[np.float64]]]:
    if params.marker_specs is not None:
        return params.marker_specs
    specs: List[Tuple[int, NDArray[np.float64]]] = []
    for i, comp in enumerate(components):
        for _name, pt in comp.markers.items():
            specs.append((i, pt))
    if not specs and len(components) > 1:
        specs.append((1, components[1].bone_segment[1]))
    return specs


def run_linkage_design(
    motion: MotionData,
    body_specs: Optional[Sequence[BodySpec]] = None,
    params: Optional[DesignParameters] = None,
    output_dir: str | Path = "./linkage_output",
    make_plots: bool = True,
    verbose: bool = True,
) -> DesignResult:
    """Run the full linkage design pipeline (with automatic caching).

    If the same motion + parameters have been run before, results are
    loaded from cache (near-instant).  Set params.force_rerun=True to
    bypass the cache.

    Pipeline:
      1. Analytic topology objective
      2. Multi-start Newton pin search
      3. Forward kinematics simulation
      4. CMA-ES global refinement
      5. Organic link centerline shaping
      6. STL + JSON export

    Parameters
    ----------
    motion : MotionData
    body_specs : list of BodySpec or None
    params : DesignParameters or None
    output_dir : str or Path
    make_plots : bool
    verbose : bool

    Returns
    -------
    DesignResult
    """
    if params is None:
        params = DesignParameters()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(params.cache_dir)

    # ---- check cache -------------------------------------------------------
    cache_key = _compute_cache_key(motion, params)
    cached = None if params.force_rerun else _load_cache(cache_dir, cache_key)

    if cached is not None:
        if verbose:
            print("=" * 58)
            print("  ✓ CACHE HIT — loading previous results (instant)")
            print("=" * 58)
            print(f"  Cache key: {cache_key}")
            print(f"  Set params.force_rerun=True to re-optimize.")

        # Reconstruct from cache
        best_candidate = _reconstruct_candidate(cached)

        # Build components
        if body_specs is None:
            scale = motion.characteristic_length
            body_specs = [
                BodySpec(name="body_a", outline=np.array([
                    [-0.1*scale, -0.08*scale], [0.3*scale, -0.08*scale],
                    [0.3*scale, 0.08*scale], [-0.1*scale, 0.08*scale]
                ]), markers={"tip": np.array([0.3*scale, 0.0])}),
                BodySpec(name="body_b", outline=np.array([
                    [-0.1*scale, -0.10*scale], [0.4*scale, -0.10*scale],
                    [0.4*scale, 0.10*scale], [-0.1*scale, 0.10*scale]
                ]), markers={"tip": np.array([0.4*scale, 0.0])}),
            ]
        components = [bs.to_rigid_component() for bs in body_specs]

        # Rebuild system & simulate for plots
        system = lto.build_two_body_linkage_system(
            components, motion.ground_a, motion.ground_b,
            best_candidate.q_a, best_candidate.q_b, best_candidate.length,
            motor_angle_function=lambda t: t,
        )
        solver = lto.ForwardKinematicsSolver(
            system, max_iterations=params.solver_max_iterations,
            tolerance=params.solver_tolerance,
        )
        simulation = solver.simulate(motion.times, motion.target_states[0])

        centerline = cached.get("centerline")
        spline_meta = cached.get("spline_metadata", {})
        if centerline is None:
            centerline, spline_meta = lto.fit_organic_link_centerline(
                states_a=motion.states_a, states_b=motion.states_b,
                q_a=best_candidate.q_a, q_b=best_candidate.q_b,
                bend_fraction=params.bend_fraction,
                tangent_fraction=params.tangent_fraction,
            )

        # Re-export STL/JSON (fabrication params may have changed)
        abs_thickness = params.fabrication_thickness * params.unit_scale
        abs_width = params.fabrication_width * params.unit_scale
        abs_hole = params.hole_radius * params.unit_scale
        stl_path = lto.export_link_stl(
            centerline,
            output_dir / f"{motion.label}_link.stl",
            width=abs_width, thickness=abs_thickness, hole_radius=abs_hole,
        )
        json_path = lto.export_design_json(
            output_dir / f"{motion.label}_design.json",
            best_candidate,
            spline_metadata=spline_meta,
            thickness=params.fabrication_thickness,
            width=params.fabrication_width,
        )
        if verbose:
            print(f"    STL  → {stl_path}")
            print(f"    JSON → {json_path}")

        # Plots
        marker_specs = _auto_marker_specs(components, params)
        if make_plots:
            lto.plot_target_motion(components, motion.target_states,
                marker=marker_specs[0] if marker_specs else (1, np.array([1.0, 0.0])))
            lto.plot_link_quality_over_cycle(
                lto.TopologyObjective(
                    states_a=motion.states_a, states_b=motion.states_b,
                    motor_positions=motion.motor_positions, gamma=params.gamma,
                    area_mode=params.area_mode,
                    characteristic_length=motion.characteristic_length,
                ), best_candidate, phases=motion.times,
            )
            lto.plot_forward_diagnostics(simulation, motion.target_states, components,
                marker_spec=marker_specs[0] if marker_specs else (1, np.array([1.0, 0.0])))
            lto.plot_singularity_profile(system, simulation)
            lto.plot_link_centerline(centerline, hole_radius=abs_hole)

        # Reconstruct candidate list (just the best one from cache)
        candidates = [best_candidate]

        return DesignResult(
            motion=motion, params=params, components=components,
            candidates=candidates, best_candidate=best_candidate,
            cma_result=None, simulation=simulation, system=system,
            centerline=centerline, spline_metadata=spline_meta,
            stl_path=stl_path, json_path=json_path, from_cache=True,
        )

    # ---- full optimization (cache miss) ------------------------------------
    if verbose:
        print("[1/5] Building analytic topology objective ...")

    if body_specs is None:
        scale = motion.characteristic_length
        body_specs = [
            BodySpec(name="body_a", outline=np.array([
                [-0.1*scale, -0.08*scale], [0.3*scale, -0.08*scale],
                [0.3*scale, 0.08*scale], [-0.1*scale, 0.08*scale]
            ]), markers={"tip": np.array([0.3*scale, 0.0])}),
            BodySpec(name="body_b", outline=np.array([
                [-0.1*scale, -0.10*scale], [0.4*scale, -0.10*scale],
                [0.4*scale, 0.10*scale], [-0.1*scale, 0.10*scale]
            ]), markers={"tip": np.array([0.4*scale, 0.0])}),
        ]
    components = [bs.to_rigid_component() for bs in body_specs]

    topology_obj = lto.TopologyObjective(
        states_a=motion.states_a, states_b=motion.states_b,
        motor_positions=motion.motor_positions,
        gamma=params.gamma, area_mode=params.area_mode,
        characteristic_length=motion.characteristic_length,
    )
    deriv_check = lto.check_topology_derivatives(
        topology_obj, np.array([0.3, 0.05, 0.8, -0.05])
    )
    if verbose:
        print(f"    Derivative check: gradient err={deriv_check['gradient_relative_error']:.2e}, "
              f"Hessian err={deriv_check['hessian_relative_error']:.2e}")

    if verbose:
        print(f"[2/5] Multi-start Newton search ({params.n_starts} starts) ...")
    candidates = lto.optimize_linkage_points(
        topology_obj, n_starts=params.n_starts, n_candidates=params.n_candidates,
        coordinate_bound=params.coordinate_bound, seed=params.seed,
        minimum_moment_arm_fraction=params.minimum_moment_arm_fraction,
    )
    if not candidates:
        raise RuntimeError("No valid candidates. Increase n_starts or relax minimum_moment_arm_fraction.")
    best = candidates[0]
    if verbose:
        print(f"    Found {len(candidates)} candidates. Best: q_a={best.q_a}, q_b={best.q_b}, "
              f"L={best.length:.3f}, CV={best.distance_cv:.2e}")

    if verbose:
        print("[3/5] Forward kinematics simulation ...")
    system = lto.build_two_body_linkage_system(
        components, motion.ground_a, motion.ground_b,
        best.q_a, best.q_b, best.length,
        motor_angle_function=lambda t: t,
    )
    solver = lto.ForwardKinematicsSolver(
        system, max_iterations=params.solver_max_iterations,
        tolerance=params.solver_tolerance,
    )
    simulation = solver.simulate(motion.times, motion.target_states[0])
    if verbose:
        conv_frac = float(np.mean(simulation.converged))
        print(f"    Converged: {conv_frac:.1%}, max residual: {simulation.residual_norms.max():.2e}")

    if verbose:
        print(f"[4/5] CMA-ES refinement ({params.cma_max_iterations} generations) ...")
    marker_specs = _auto_marker_specs(components, params)
    global_obj = lto.GlobalLinkageObjective(
        components=components, target_states=motion.target_states,
        times=motion.times, ground_a=motion.ground_a, ground_b=motion.ground_b,
        marker_specs=marker_specs,
        bone_segments=[c.bone_segment for c in components],
        weights=lto.GlobalWeights(marker=1.0, state=0.2, joint=0.05, singular=1e-4, failure=1e5),
        characteristic_length=motion.characteristic_length,
        time_stride=params.cma_time_stride,
        solver_kwargs={"max_iterations": params.solver_max_iterations, "tolerance": params.solver_tolerance},
    )
    cma_result = lto.run_cma_es(
        global_obj, initial_parameters=best.parameters,
        sigma0=params.cma_sigma0, max_iterations=params.cma_max_iterations,
        population_size=params.cma_population_size, seed=params.seed + 4,
        verbose=False,
    )
    if verbose:
        print(f"    Best cost: {cma_result.best_cost:.6e}, params: {cma_result.best_parameters}")

    refined_z = cma_result.best_parameters
    refined_q_a, refined_q_b = refined_z[:2], refined_z[2:]
    refined_metrics = topology_obj.candidate_metrics(refined_z)
    refined_ev = topology_obj.evaluate(refined_z, derivatives=False)
    refined_candidate = lto.TopologyCandidate(
        q_a=refined_q_a.copy(), q_b=refined_q_b.copy(),
        length=refined_metrics["length"], objective=refined_ev.total,
        distance_variance=refined_ev.distance_variance,
        distance_cv=refined_metrics["distance_cv"],
        minimum_moment_arm=refined_metrics["minimum_moment_arm"],
        percentile05_moment_arm=refined_metrics["percentile05_moment_arm"],
        mean_moment_arm=refined_metrics["mean_moment_arm"],
        area_energy=refined_ev.area_energy,
        converged=True, iterations=0,
        start=best.parameters.copy(), history=[],
    )

    refined_sim = lto.simulate_candidate(
        components, motion.target_states, motion.times,
        refined_candidate, motion.ground_a, motion.ground_b,
        solver_kwargs={"max_iterations": params.solver_max_iterations, "tolerance": params.solver_tolerance},
    )

    if verbose:
        print("[5/5] Shaping organic link + exporting STL/JSON ...")
    centerline, spline_meta = lto.fit_organic_link_centerline(
        states_a=motion.states_a, states_b=motion.states_b,
        q_a=refined_q_a, q_b=refined_q_b,
        bend_fraction=params.bend_fraction,
        tangent_fraction=params.tangent_fraction,
    )

    abs_thickness = params.fabrication_thickness * params.unit_scale
    abs_width = params.fabrication_width * params.unit_scale
    abs_hole = params.hole_radius * params.unit_scale
    stl_path = lto.export_link_stl(
        centerline, output_dir / f"{motion.label}_link.stl",
        width=abs_width, thickness=abs_thickness, hole_radius=abs_hole,
    )
    json_path = lto.export_design_json(
        output_dir / f"{motion.label}_design.json",
        refined_candidate, spline_metadata=spline_meta,
        thickness=params.fabrication_thickness, width=params.fabrication_width,
    )
    if verbose:
        print(f"    STL  → {stl_path}")
        print(f"    JSON → {json_path}")

    # ---- save cache -------------------------------------------------------
    cache_data = {
        "q_a": refined_q_a.tolist(),
        "q_b": refined_q_b.tolist(),
        "length": refined_candidate.length,
        "objective": refined_candidate.objective,
        "distance_variance": refined_candidate.distance_variance,
        "distance_cv": refined_candidate.distance_cv,
        "minimum_moment_arm": refined_candidate.minimum_moment_arm,
        "percentile05_moment_arm": refined_candidate.percentile05_moment_arm,
        "mean_moment_arm": refined_candidate.mean_moment_arm,
        "area_energy": refined_candidate.area_energy,
        "converged": refined_candidate.converged,
        "iterations": refined_candidate.iterations,
        "start": best.parameters.tolist(),
        "centerline": centerline,
        "spline_metadata": spline_meta,
    }
    _save_cache(cache_dir, cache_key, cache_data)
    if verbose:
        print(f"    Cached → {cache_dir / cache_key}.npz")

    # ---- plots ------------------------------------------------------------
    if make_plots:
        lto.plot_target_motion(components, motion.target_states,
            marker=marker_specs[0] if marker_specs else (1, np.array([1.0, 0.0])))
        lto.plot_candidate_metrics(candidates)
        lto.plot_link_quality_over_cycle(topology_obj, refined_candidate, phases=motion.times)
        lto.plot_forward_diagnostics(refined_sim, motion.target_states, components,
            marker_spec=marker_specs[0] if marker_specs else (1, np.array([1.0, 0.0])))
        lto.plot_singularity_profile(system, refined_sim)
        lto.plot_link_centerline(centerline, hole_radius=abs_hole)

    return DesignResult(
        motion=motion, params=params, components=components,
        candidates=candidates, best_candidate=refined_candidate,
        cma_result=cma_result, simulation=refined_sim, system=system,
        centerline=centerline, spline_metadata=spline_meta,
        stl_path=stl_path, json_path=json_path, from_cache=False,
    )


def quick_design(
    motion_data: MotionData,
    ground_a: ArrayLike = (0.0, 0.0),
    ground_b: ArrayLike = (80.0, 0.0),
    motor_axis: Optional[ArrayLike] = None,
    output_dir: str | Path = "./linkage_output",
    **param_overrides,
) -> DesignResult:
    """One-call convenience: design a linkage from motion data.

    Parameters
    ----------
    motion_data : MotionData
    ground_a, ground_b : (2,) array_like
    motor_axis : (2,) array_like or None
    output_dir : str or Path
    **param_overrides
        Any DesignParameters field, e.g. n_starts=120, fabrication_thickness=4.0.
    """
    ga = np.asarray(ground_a, dtype=float).ravel()
    gb = np.asarray(ground_b, dtype=float).ravel()
    motion_data.ground_a = ga
    motion_data.ground_b = gb
    if motor_axis is not None:
        ma = np.asarray(motor_axis, dtype=float).ravel()
        motion_data.motor_positions = np.repeat(ma[None, :], motion_data.n_frames, axis=0)

    defaults = {f.name: getattr(DesignParameters(), f.name)
                for f in DesignParameters.__dataclass_fields__.values()}
    defaults.update(param_overrides)
    params = DesignParameters(**defaults)
    return run_linkage_design(motion_data, params=params, output_dir=output_dir)


# ---------------------------------------------------------------------------
# Multi-link utilities
# ---------------------------------------------------------------------------

def build_multi_link_trajectories(
    designs: Dict[str, Tuple[MotionData, DesignResult]],
) -> Dict[str, NDArray[np.float64]]:
    """Build link-segment trajectories for collision detection across links."""
    trajectories: Dict[str, NDArray[np.float64]] = {}
    for name, (motion, result) in designs.items():
        segments = np.asarray([
            [lto.world_point(sa, result.best_candidate.q_a),
             lto.world_point(sb, result.best_candidate.q_b)]
            for sa, sb in zip(motion.states_a, motion.states_b)
        ])
        trajectories[name] = segments
    return trajectories


def assign_multi_link_layers(
    designs: Dict[str, Tuple[MotionData, DesignResult]],
    layer_spacing_mm: float = 2.0,
) -> Dict[str, Dict[str, float]]:
    """Build intersection graph and assign Z layers."""
    traj = build_multi_link_trajectories(designs)
    graph = lto.build_intersection_graph(traj)
    return lto.assign_depth_layers(graph, layer_spacing=layer_spacing_mm)


def export_all_stls(
    designs: Dict[str, Tuple[MotionData, DesignResult]],
    output_dir: str | Path,
    layer_spacing_mm: float = 2.0,
    thickness_mm: float = 3.0,
    width_mm: float = 5.0,
    hole_radius_mm: float = 2.0,
) -> Dict[str, Path]:
    """Export STLs for multiple links with collision Z-layering.

    Parameters
    ----------
    designs : dict of link_name → (MotionData, DesignResult)
    output_dir : str or Path
    layer_spacing_mm : float
        Z spacing between layers in mm.
    thickness_mm, width_mm, hole_radius_mm : float
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    layers = assign_multi_link_layers(designs, layer_spacing_mm=layer_spacing_mm)
    paths: Dict[str, Path] = {}
    for name, (motion, result) in designs.items():
        layer = layers.get(name, {"layer": 0, "z_offset": 0.0})
        unit_scale = result.params.unit_scale
        stl_path = lto.export_link_stl(
            result.centerline, output_dir / f"{name}.stl",
            width=width_mm * unit_scale,
            thickness=thickness_mm * unit_scale,
            hole_radius=hole_radius_mm * unit_scale,
            layer_z=layer["z_offset"] * unit_scale,
        )
        paths[name] = stl_path
    return paths


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def run_example(output_dir: str | Path = "./example_output") -> DesignResult:
    """Run a complete example using a synthetic crank-rocker motion."""
    print("=" * 58)
    print("  REAL-WORLD LINKAGE DESIGNER — EXAMPLE RUN")
    print("=" * 58)
    motion = create_crank_rocker_motion(
        ground_a=(0.0, 0.0), ground_b=(80.0, 0.0),
        crank_radius=25.0, rocker_radius=55.0, coupler_length=62.5,
        n_samples=121,
    )
    body_specs = [
        BodySpec(name="crank", outline=np.array([
            [0.0, -2.5], [25.0, -2.5], [25.0, 2.5], [0.0, 2.5]
        ]), markers={"tip": np.array([25.0, 0.0])}),
        BodySpec(name="rocker", outline=np.array([
            [0.0, -3.0], [55.0, -3.0], [55.0, 3.0], [0.0, 3.0]
        ]), markers={"tip": np.array([55.0, 0.0])}),
    ]
    params = DesignParameters(
        n_starts=80, n_candidates=8, cma_max_iterations=30, cma_time_stride=3,
        fabrication_thickness=3.0, fabrication_width=5.0, hole_radius=2.0,
        unit_scale=1.0,
    )
    result = run_linkage_design(
        motion=motion, body_specs=body_specs, params=params,
        output_dir=output_dir, make_plots=True, verbose=True,
    )
    print("\n" + result.summary())
    return result


if __name__ == "__main__":
    run_example()
