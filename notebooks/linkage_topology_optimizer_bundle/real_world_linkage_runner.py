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
    ...     ground_b=(80.0, 0.0),   # mm
    ...     motor_axis=(80.0, 0.0),  # same as ground_b for rocker-type
    ...     output_dir="./my_linkage_output",
    ... )

For custom motion, see:
    - create_motion_from_arrays()
    - load_motion_from_csv()
    - create_crank_rocker_motion()

All internal computation uses abstract units; the `unit_scale` parameter
(units/mm) maps them to real-world millimeters for fabrication output.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Any
import json
import math

import numpy as np
from numpy.typing import ArrayLike, NDArray

# Re-use the research optimizer
import linkage_topology_optimizer as lto


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
        World position of the motor / input pivot per frame.
        For a crank, this is the crank ground pivot (often constant).
        For a rocker output, this is the rocker ground pivot.
    ground_a : (2,) array
        Ground pivot of body A (constant).
    ground_b : (2,) array
        Ground pivot of body B (constant).
    times : (n_frames,) array or None
        Phase/time values. If None, linspace(0, 2π, n_frames) is used.
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
        # validate
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
        """Stacked array shaped (n_frames, 2, 3) for use with the optimizer."""
        return np.stack([self.states_a, self.states_b], axis=1)

    @property
    def characteristic_length(self) -> float:
        """Auto-computed spatial scale of the motion."""
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
        Label like "crank" or "rocker".
    outline : (n_vertices, 2) array
        Polygon vertices of the body in local coordinates.
        If None, a default rectangle is created.
    bone_segment : ((2,), (2,))  or None
        The body's "bone" as (start, end) in local coords.  Defaults to
        [(0,0), (1,0)] scaled to outline length.
    markers : dict[str, (2,)]
        Named marker points in local coords, e.g. {"tip": [0.8, 0.0]}.
    """
    name: str
    outline: Optional[NDArray[np.float64]] = None
    bone_segment: Optional[Tuple[NDArray[np.float64], NDArray[np.float64]]] = None
    markers: Dict[str, NDArray[np.float64]] = field(default_factory=dict)

    def __post_init__(self):
        if self.outline is None:
            self.outline = np.array([
                [-0.1, -0.1], [1.0, -0.1], [1.0, 0.1], [-0.1, 0.1]
            ], dtype=float)
        else:
            self.outline = np.asarray(self.outline, dtype=float)
        if self.bone_segment is None:
            xs = self.outline[:, 0]
            self.bone_segment = (
                np.array([xs.min(), 0.0]),
                np.array([xs.max(), 0.0]),
            )
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
    """All tunable knobs for the optimization and fabrication pipeline.

    Parameters
    ----------
    n_samples : int
        Number of time samples for the motion cycle (default 121).
    n_starts : int
        Multi-start Newton search count. Higher = more thorough (default 80).
    n_candidates : int
        Number of distinct candidates to keep after clustering (default 8).
    gamma : float
        Weight of the area/moment-arm penalty term (default 0.1).
    coordinate_bound : float
        ±bound for pin coordinates in abstract units (default None → auto).
    area_mode : str
        "barrier" (recommended) or "paper" for the area penalty.
    cma_max_iterations : int
        CMA-ES generations for global refinement (default 50).
    cma_population_size : int or None
        CMA-ES population size (default None → auto).
    cma_time_stride : int
        Subsample factor for CMA-ES speed (default 3; set 1 for final).
    cma_sigma0 : float
        Initial CMA-ES step size (default 0.10).
    fabrication_thickness : float
        Link thickness in mm for STL export (default 3.0).
    fabrication_width : float
        Link width in mm for STL export (default 5.0).
    hole_radius : float
        Pin hole radius in mm for STL export (default 2.0).
    unit_scale : float
        How many abstract units per millimeter.  E.g., if your motion data
        uses mm units, set unit_scale=1.0.  If your motion uses meters,
        set unit_scale=1000.0 (1000 mm per meter).  Default 1.0.
    bend_fraction : float
        How much the organic link bends, as fraction of link length
        (default 0.08).
    tangent_fraction : float
        How far the tangent handles extend, as fraction of link length
        (default 0.18).
    marker_specs : list of (body_index, local_point)
        Which markers to track during CMA-ES refinement.
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

    # fabrication
    fabrication_thickness: float = 3.0   # mm
    fabrication_width: float = 5.0        # mm
    hole_radius: float = 2.0              # mm
    unit_scale: float = 1.0               # abstract-units / mm

    # organic shaping
    bend_fraction: float = 0.08
    tangent_fraction: float = 0.18

    # marker specs for CMA-ES (list of (body_index, local_point))
    marker_specs: Optional[List[Tuple[int, NDArray[np.float64]]]] = None

    # solver knobs
    solver_max_iterations: int = 40
    solver_tolerance: float = 1e-9


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

    def summary(self) -> str:
        bc = self.best_candidate
        lines = [
            "=" * 58,
            "  LINKAGE DESIGN RESULT",
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
    states_a : (n_frames, 3) array_like
        Body A states [theta, x, y].
    states_b : (n_frames, 3) array_like
        Body B states [theta, x, y].
    ground_a : (2,) array_like
        Ground pivot for body A.
    ground_b : (2,) array_like
        Ground pivot for body B.
    motor_positions : (n_frames, 2) array_like or None
        Motor/input world positions. If None, uses ground_b repeated.
    times : (n_frames,) array_like or None
    label : str
    """
    sa = np.asarray(states_a, dtype=float)
    sb = np.asarray(states_b, dtype=float)
    n = len(sa)
    if motor_positions is None:
        mp = np.repeat(np.asarray(ground_b, dtype=float)[None, :], n, axis=0)
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

    This is a convenience for testing.  For real-world use, import your
    own motion data via create_motion_from_arrays() or load_motion_from_csv().

    Parameters
    ----------
    ground_a, ground_b : (2,) array_like
        Ground pivot positions.
    crank_radius : float
        Length of the crank (body A).
    rocker_radius : float
        Length of the rocker (body B).
    coupler_length : float
        Length of the hidden coupler connecting crank-tip to rocker-tip.
        The optimizer will try to rediscover this from motion alone.
    n_samples : int
        Number of time samples.
    phase_offset : float
        Initial crank angle offset (radians).
    initial_branch : "upper" or "lower"
        Which assembly branch.

    Returns
    -------
    MotionData
    """
    times = np.linspace(0.0, 2.0 * np.pi, n_samples, endpoint=False)
    oa = np.asarray(ground_a, dtype=float)
    ob = np.asarray(ground_b, dtype=float)
    states = np.zeros((n_samples, 2, 3), dtype=float)
    previous = None

    for i, t in enumerate(times):
        theta_a = float(t + phase_offset)
        pa = oa + lto.rotation(theta_a) @ np.array([crank_radius, 0.0])

        # two-circle intersection
        delta = ob - pa
        d = float(np.linalg.norm(delta))
        if d < 1e-12 or d > coupler_length + rocker_radius + 1e-10 or d < abs(coupler_length - rocker_radius) - 1e-10:
            raise ValueError(f"The crank-rocker dimensions do not close at phase {t:.3f}.")
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
        states_a=states[:, 0],
        states_b=states[:, 1],
        motor_positions=motor_positions,
        ground_a=oa, ground_b=ob,
        times=times,
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

    Each CSV should contain rows of [theta, x, y] (or reorderable via columns).
    Both CSVs must have the same number of rows.

    Parameters
    ----------
    path_a, path_b : str or Path
        Paths to CSV files for body A and body B.
    ground_a, ground_b : (2,) array_like
    motor_positions_path : str, Path, or None
        CSV for motor positions (2 columns: x, y). If None, ground_a is used.
    delimiter : str
    skip_header : int
        Rows to skip at top of CSV.
    columns : (col_theta, col_x, col_y)
        Zero-based column indices in the CSV.
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


def load_motion_from_json(
    path: str | Path,
    label: str = "json_motion",
) -> MotionData:
    """Load motion from a JSON file.

    Expected format:
    {
        "states_a": [[theta, x, y], ...],   // n_frames × 3
        "states_b": [[theta, x, y], ...],
        "ground_a": [x, y],
        "ground_b": [x, y],
        "motor_positions": [[x, y], ...],    // optional; n_frames × 2
        "times": [t0, t1, ...]               // optional
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
# Pipeline
# ---------------------------------------------------------------------------

def _auto_marker_specs(
    components: List[lto.RigidComponent],
    params: DesignParameters,
) -> List[Tuple[int, NDArray[np.float64]]]:
    """If user didn't provide marker_specs, derive sensible defaults."""
    if params.marker_specs is not None:
        return params.marker_specs
    specs: List[Tuple[int, NDArray[np.float64]]] = []
    for i, comp in enumerate(components):
        for name, pt in comp.markers.items():
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
    """Run the full linkage design pipeline.

    1. Build the analytic topology objective.
    2. Multi-start Newton search for pin locations.
    3. Forward kinematics simulation of the best candidate.
    4. CMA-ES global refinement.
    5. Organic link centerline shaping.
    6. STL + JSON export.

    Parameters
    ----------
    motion : MotionData
        Target motion to reproduce.
    body_specs : list of BodySpec or None
        Geometry for each body. If None, defaults are created based on motion scale.
    params : DesignParameters or None
        All tunable parameters. If None, defaults are used.
    output_dir : str or Path
        Where to write STL and JSON files.
    make_plots : bool
        Whether to show diagnostic plots.
    verbose : bool
        Whether to print progress.

    Returns
    -------
    DesignResult
    """
    if params is None:
        params = DesignParameters()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- build components -------------------------------------------------
    if body_specs is None:
        scale = motion.characteristic_length
        body_specs = [
            BodySpec(
                name="body_a",
                outline=np.array([
                    [-0.1 * scale, -0.08 * scale],
                    [0.3 * scale, -0.08 * scale],
                    [0.3 * scale, 0.08 * scale],
                    [-0.1 * scale, 0.08 * scale],
                ]),
                markers={"tip": np.array([0.3 * scale, 0.0])},
            ),
            BodySpec(
                name="body_b",
                outline=np.array([
                    [-0.1 * scale, -0.10 * scale],
                    [0.4 * scale, -0.10 * scale],
                    [0.4 * scale, 0.10 * scale],
                    [-0.1 * scale, 0.10 * scale],
                ]),
                markers={"tip": np.array([0.4 * scale, 0.0])},
            ),
        ]

    components = [bs.to_rigid_component() for bs in body_specs]

    # ---- analytic topology optimization -----------------------------------
    if verbose:
        print("[1/5] Building analytic topology objective ...")
    topology_obj = lto.TopologyObjective(
        states_a=motion.states_a,
        states_b=motion.states_b,
        motor_positions=motion.motor_positions,
        gamma=params.gamma,
        area_mode=params.area_mode,
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
        topology_obj,
        n_starts=params.n_starts,
        n_candidates=params.n_candidates,
        coordinate_bound=params.coordinate_bound,
        seed=params.seed,
        minimum_moment_arm_fraction=params.minimum_moment_arm_fraction,
    )
    if not candidates:
        raise RuntimeError(
            "No valid candidates found. Try increasing n_starts or "
            "relaxing minimum_moment_arm_fraction."
        )
    best = candidates[0]
    if verbose:
        print(f"    Found {len(candidates)} candidates. Best: q_a={best.q_a}, q_b={best.q_b}, "
              f"L={best.length:.3f}, CV={best.distance_cv:.2e}")

    # ---- forward simulation -----------------------------------------------
    if verbose:
        print("[3/5] Forward kinematics simulation ...")
    system = lto.build_two_body_linkage_system(
        components,
        motion.ground_a, motion.ground_b,
        best.q_a, best.q_b, best.length,
        motor_angle_function=lambda t: t,
    )
    solver = lto.ForwardKinematicsSolver(
        system,
        max_iterations=params.solver_max_iterations,
        tolerance=params.solver_tolerance,
    )
    simulation = solver.simulate(motion.times, motion.target_states[0])
    conv_frac = float(np.mean(simulation.converged))
    if verbose:
        print(f"    Converged: {conv_frac:.1%}, max residual: {simulation.residual_norms.max():.2e}")

    # ---- CMA-ES refinement ------------------------------------------------
    if verbose:
        print(f"[4/5] CMA-ES refinement ({params.cma_max_iterations} generations) ...")
    marker_specs = _auto_marker_specs(components, params)
    global_obj = lto.GlobalLinkageObjective(
        components=components,
        target_states=motion.target_states,
        times=motion.times,
        ground_a=motion.ground_a,
        ground_b=motion.ground_b,
        marker_specs=marker_specs,
        bone_segments=[c.bone_segment for c in components],
        weights=lto.GlobalWeights(marker=1.0, state=0.2, joint=0.05, singular=1e-4, failure=1e5),
        characteristic_length=motion.characteristic_length,
        time_stride=params.cma_time_stride,
        solver_kwargs={"max_iterations": params.solver_max_iterations, "tolerance": params.solver_tolerance},
    )

    cma_result = lto.run_cma_es(
        global_obj,
        initial_parameters=best.parameters,
        sigma0=params.cma_sigma0,
        max_iterations=params.cma_max_iterations,
        population_size=params.cma_population_size,
        seed=params.seed + 4,
        verbose=False,
    )
    if verbose:
        print(f"    Best cost: {cma_result.best_cost:.6e}, "
              f"params: {cma_result.best_parameters}")

    # update best candidate with CMA-ES result
    refined_z = cma_result.best_parameters
    refined_q_a, refined_q_b = refined_z[:2], refined_z[2:]
    refined_metrics = topology_obj.candidate_metrics(refined_z)
    refined_ev = topology_obj.evaluate(refined_z, derivatives=False)
    refined_candidate = lto.TopologyCandidate(
        q_a=refined_q_a.copy(), q_b=refined_q_b.copy(),
        length=refined_metrics["length"],
        objective=refined_ev.total,
        distance_variance=refined_ev.distance_variance,
        distance_cv=refined_metrics["distance_cv"],
        minimum_moment_arm=refined_metrics["minimum_moment_arm"],
        percentile05_moment_arm=refined_metrics["percentile05_moment_arm"],
        mean_moment_arm=refined_metrics["mean_moment_arm"],
        area_energy=refined_ev.area_energy,
        converged=True,
        iterations=0,
        start=best.parameters.copy(),
        history=[],
    )

    # re-simulate with refined params
    refined_sim = lto.simulate_candidate(
        components, motion.target_states, motion.times,
        refined_candidate, motion.ground_a, motion.ground_b,
        solver_kwargs={"max_iterations": params.solver_max_iterations, "tolerance": params.solver_tolerance},
    )

    # ---- organic link shape + export --------------------------------------
    if verbose:
        print("[5/5] Shaping organic link + exporting STL/JSON ...")

    centerline, spline_meta = lto.fit_organic_link_centerline(
        states_a=motion.states_a,
        states_b=motion.states_b,
        q_a=refined_q_a,
        q_b=refined_q_b,
        bend_fraction=params.bend_fraction,
        tangent_fraction=params.tangent_fraction,
    )

    # Convert fabrication params from mm to abstract units for STL export
    abs_thickness = params.fabrication_thickness * params.unit_scale
    abs_width = params.fabrication_width * params.unit_scale
    abs_hole = params.hole_radius * params.unit_scale

    stl_path = lto.export_link_stl(
        centerline,
        output_dir / f"{motion.label}_link.stl",
        width=abs_width,
        thickness=abs_thickness,
        hole_radius=abs_hole,
    )
    json_path = lto.export_design_json(
        output_dir / f"{motion.label}_design.json",
        refined_candidate,
        spline_metadata=spline_meta,
        thickness=params.fabrication_thickness,
        width=params.fabrication_width,
    )
    if verbose:
        print(f"    STL  → {stl_path}")
        print(f"    JSON → {json_path}")

    # ---- plots ------------------------------------------------------------
    if make_plots:
        lto.plot_target_motion(components, motion.target_states, marker=marker_specs[0] if marker_specs else (1, np.array([1.0, 0.0])))
        lto.plot_candidate_metrics(candidates)
        lto.plot_link_quality_over_cycle(topology_obj, refined_candidate, phases=motion.times)
        lto.plot_forward_diagnostics(refined_sim, motion.target_states, components, marker_spec=marker_specs[0] if marker_specs else (1, np.array([1.0, 0.0])))
        lto.plot_singularity_profile(system, refined_sim)
        lto.plot_link_centerline(centerline, hole_radius=abs_hole)

    return DesignResult(
        motion=motion,
        params=params,
        components=components,
        candidates=candidates,
        best_candidate=refined_candidate,
        cma_result=cma_result,
        simulation=refined_sim,
        system=system,
        centerline=centerline,
        spline_metadata=spline_meta,
        stl_path=stl_path,
        json_path=json_path,
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
        Pre-built motion data (use create_motion_from_arrays, etc.)
    ground_a, ground_b : (2,) array_like
        Ground pivot positions (overrides those in motion_data if provided).
    motor_axis : (2,) array_like or None
        Motor axis world position.  If None, uses ground_a.
        This is the point that forms the area triangle with the two pin
        positions.  Typically the input crank ground pivot.
    output_dir : str or Path
    **param_overrides
        Any DesignParameters field, e.g. n_starts=120, fabrication_thickness=4.0.

    Returns
    -------
    DesignResult
    """
    ga = np.asarray(ground_a, dtype=float).ravel()
    gb = np.asarray(ground_b, dtype=float).ravel()
    motion_data.ground_a = ga
    motion_data.ground_b = gb

    if motor_axis is not None:
        ma = np.asarray(motor_axis, dtype=float).ravel()
        motion_data.motor_positions = np.repeat(ma[None, :], motion_data.n_frames, axis=0)

    params_dict = {f.name: getattr(DesignParameters(), f.name) for f in DesignParameters.__dataclass_fields__.values()}
    params_dict.update(param_overrides)
    params = DesignParameters(**params_dict)

    return run_linkage_design(motion_data, params=params, output_dir=output_dir)


# ---------------------------------------------------------------------------
# Multi-link utilities (for future expansion)
# ---------------------------------------------------------------------------

def build_multi_link_trajectories(
    designs: Dict[str, Tuple[MotionData, DesignResult]],
) -> Dict[str, NDArray[np.float64]]:
    """Build link-segment trajectories for multiple designed links.

    Each entry maps a link name → (n_frames, 2, 2) array of (start, end)
    world points for collision detection.
    """
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
    """Build intersection graph and assign Z layers for multiple links."""
    traj = build_multi_link_trajectories(designs)
    graph = lto.build_intersection_graph(traj)
    layers = lto.assign_depth_layers(graph, layer_spacing=layer_spacing_mm)
    return layers


# ---------------------------------------------------------------------------
# Export helpers
# ---------------------------------------------------------------------------

def export_all_stls(
    designs: Dict[str, Tuple[MotionData, DesignResult]],
    output_dir: str | Path,
    layer_spacing_mm: float = 2.0,
    thickness_mm: float = 3.0,
    width_mm: float = 5.0,
    hole_radius_mm: float = 2.0,
) -> Dict[str, Path]:
    """Export STL files for multiple links with collision-layer Z offsets.

    Parameters
    ----------
    designs : dict
        Mapping of link_name → (MotionData, DesignResult).
    output_dir : str or Path
    layer_spacing_mm : float
        Z spacing between layers in mm.
    thickness_mm, width_mm, hole_radius_mm : float
        Fabrication parameters in mm.

    Returns
    -------
    dict of link_name → STL Path
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    layers = assign_multi_link_layers(designs, layer_spacing_mm=layer_spacing_mm)

    paths: Dict[str, Path] = {}
    for name, (motion, result) in designs.items():
        layer = layers.get(name, {"layer": 0, "z_offset": 0.0})
        unit_scale = result.params.unit_scale
        stl_path = lto.export_link_stl(
            result.centerline,
            output_dir / f"{name}.stl",
            width=width_mm * unit_scale,
            thickness=thickness_mm * unit_scale,
            hole_radius=hole_radius_mm * unit_scale,
            layer_z=layer["z_offset"] * unit_scale,
        )
        paths[name] = stl_path
    return paths


# ---------------------------------------------------------------------------
# Example / demo
# ---------------------------------------------------------------------------

def run_example(output_dir: str | Path = "./example_output") -> DesignResult:
    """Run a complete example using a synthetic crank-rocker motion.

    This demonstrates the full pipeline end-to-end and serves as a
    starting point for custom designs.
    """
    print("=" * 58)
    print("  REAL-WORLD LINKAGE DESIGNER — EXAMPLE RUN")
    print("=" * 58)

    motion = create_crank_rocker_motion(
        ground_a=(0.0, 0.0),
        ground_b=(80.0, 0.0),
        crank_radius=25.0,
        rocker_radius=55.0,
        coupler_length=62.5,
        n_samples=121,
    )

    body_specs = [
        BodySpec(
            name="crank",
            outline=np.array([
                [0.0, -2.5], [25.0, -2.5], [25.0, 2.5], [0.0, 2.5]
            ]),
            markers={"tip": np.array([25.0, 0.0])},
        ),
        BodySpec(
            name="rocker",
            outline=np.array([
                [0.0, -3.0], [55.0, -3.0], [55.0, 3.0], [0.0, 3.0]
            ]),
            markers={"tip": np.array([55.0, 0.0])},
        ),
    ]

    params = DesignParameters(
        n_samples=121,
        n_starts=80,
        n_candidates=8,
        gamma=0.1,
        coordinate_bound=None,
        cma_max_iterations=30,
        cma_time_stride=3,
        fabrication_thickness=3.0,
        fabrication_width=5.0,
        hole_radius=2.0,
        unit_scale=1.0,
    )

    result = run_linkage_design(
        motion=motion,
        body_specs=body_specs,
        params=params,
        output_dir=output_dir,
        make_plots=True,
        verbose=True,
    )

    print()
    print(result.summary())
    return result


if __name__ == "__main__":
    run_example()
