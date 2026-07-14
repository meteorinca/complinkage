"""
Linkage Topology and Kinematic Optimization
===========================================

A research-grade, Jupyter-friendly prototype for:

1. Sparse Newton/Gauss-Newton forward kinematics of planar rigid bodies.
2. Analytic multi-start optimization of two linkage anchor points.
3. Derivative-free CMA-ES refinement with trajectory, state, alignment,
   and singularity costs.
4. Diagnostic plots, interactive phase/candidate widgets, animation.
5. Organic centerline shaping, collision-layer assignment, and STL export.

State convention
----------------
Each planar rigid component has state [theta, x, y], where theta is radians
and [x, y] is the world position of the component's local origin.

This is an engineering prototype, not a substitute for tolerance analysis,
stress analysis, bearing selection, backlash analysis, or physical testing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Any
import json
import math
import warnings

import numpy as np
from numpy.typing import ArrayLike, NDArray

import scipy.sparse as sp
from scipy.sparse.linalg import spsolve, svds

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Polygon


FloatArray = NDArray[np.float64]


# -----------------------------------------------------------------------------
# Basic planar rigid-body mathematics
# -----------------------------------------------------------------------------

def rotation(theta: float) -> FloatArray:
    """2x2 planar rotation matrix."""
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=float)


def drotation(theta: float) -> FloatArray:
    """Derivative dR/dtheta."""
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[-s, -c], [c, -s]], dtype=float)


def wrap_angle(theta: ArrayLike) -> FloatArray:
    """Map angle(s) to [-pi, pi)."""
    x = np.asarray(theta, dtype=float)
    return (x + np.pi) % (2.0 * np.pi) - np.pi


def circular_mean(angles: ArrayLike) -> float:
    """Circular mean of angles in radians."""
    a = np.asarray(angles, dtype=float)
    return float(math.atan2(np.mean(np.sin(a)), np.mean(np.cos(a))))


def world_point(state: ArrayLike, q_local: ArrayLike) -> FloatArray:
    """Transform a local point q into world coordinates."""
    s = np.asarray(state, dtype=float)
    q = np.asarray(q_local, dtype=float)
    return rotation(float(s[0])) @ q + s[1:3]


def point_state_jacobian(state: ArrayLike, q_local: ArrayLike) -> FloatArray:
    """Jacobian d(world_point)/d[theta,x,y], shape (2,3)."""
    s = np.asarray(state, dtype=float)
    q = np.asarray(q_local, dtype=float)
    j = np.zeros((2, 3), dtype=float)
    j[:, 0] = drotation(float(s[0])) @ q
    j[:, 1:] = np.eye(2)
    return j


def point_to_segment_distance(point: ArrayLike, a: ArrayLike, b: ArrayLike) -> float:
    """Euclidean distance from point to a closed line segment."""
    p = np.asarray(point, dtype=float)
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    ab = b - a
    denom = float(ab @ ab)
    if denom < 1e-15:
        return float(np.linalg.norm(p - a))
    tau = float(np.clip(((p - a) @ ab) / denom, 0.0, 1.0))
    return float(np.linalg.norm(p - (a + tau * ab)))


def finite_or(value: float, fallback: float) -> float:
    return float(value) if np.isfinite(value) else float(fallback)


# -----------------------------------------------------------------------------
# Data model
# -----------------------------------------------------------------------------
@dataclass
class RigidComponent:
    name: str
    outline: FloatArray = field(
        default_factory=lambda: np.array(
            [[-0.1, -0.08], [1.0, -0.08], [1.0, 0.08], [-0.1, 0.08]],
            dtype=float,
        )
    )
    bone_segment: Tuple[FloatArray, FloatArray] = field(
        default_factory=lambda: (np.array([0.0, 0.0]), np.array([1.0, 0.0]))
    )
    markers: Dict[str, FloatArray] = field(default_factory=dict)

    def transformed_outline(self, state: ArrayLike) -> FloatArray:
        s = np.asarray(state, dtype=float)
        return (rotation(float(s[0])) @ self.outline.T).T + s[1:3]


class Constraint:
    """Base class for residual constraints C(s,t)=0."""

    dimension: int = 0

    def residual_and_blocks(
        self, states: FloatArray, t: float
    ) -> Tuple[FloatArray, Dict[int, FloatArray]]:
        raise NotImplementedError


@dataclass
class GroundPointConstraint(Constraint):
    component: int
    local_point: FloatArray
    target_world: FloatArray
    dimension: int = 2

    def residual_and_blocks(self, states: FloatArray, t: float):
        p = world_point(states[self.component], self.local_point)
        r = p - self.target_world
        return r, {self.component: point_state_jacobian(states[self.component], self.local_point)}


@dataclass
class MotorAngleConstraint(Constraint):
    component: int
    angle_function: Callable[[float], float]
    dimension: int = 1

    def residual_and_blocks(self, states: FloatArray, t: float):
        target = float(self.angle_function(t))
        r = np.array([float(wrap_angle(states[self.component, 0] - target))])
        j = np.array([[1.0, 0.0, 0.0]], dtype=float)
        return r, {self.component: j}


@dataclass
class PinConstraint(Constraint):
    component_a: int
    local_a: FloatArray
    component_b: int
    local_b: FloatArray
    dimension: int = 2

    def residual_and_blocks(self, states: FloatArray, t: float):
        pa = world_point(states[self.component_a], self.local_a)
        pb = world_point(states[self.component_b], self.local_b)
        ja = point_state_jacobian(states[self.component_a], self.local_a)
        jb = point_state_jacobian(states[self.component_b], self.local_b)
        return pa - pb, {self.component_a: ja, self.component_b: -jb}


@dataclass
class DistanceConstraint(Constraint):
    component_a: int
    local_a: FloatArray
    component_b: int
    local_b: FloatArray
    length: float
    dimension: int = 1

    def residual_and_blocks(self, states: FloatArray, t: float):
        pa = world_point(states[self.component_a], self.local_a)
        pb = world_point(states[self.component_b], self.local_b)
        d = pa - pb
        scale = max(abs(float(self.length)), 1e-9)
        # Smooth squared-distance residual with approximately length units.
        r = np.array([0.5 * (float(d @ d) - self.length**2) / scale])
        ja = point_state_jacobian(states[self.component_a], self.local_a)
        jb = point_state_jacobian(states[self.component_b], self.local_b)
        return r, {
            self.component_a: (d @ ja / scale).reshape(1, 3),
            self.component_b: (-d @ jb / scale).reshape(1, 3),
        }


@dataclass
class ConstraintSystem:
    components: Sequence[RigidComponent]
    constraints: Sequence[Constraint]

    @property
    def n_components(self) -> int:
        return len(self.components)

    @property
    def n_dof(self) -> int:
        return 3 * self.n_components

    @property
    def n_residual(self) -> int:
        return int(sum(c.dimension for c in self.constraints))

    def residual_jacobian(self, states: FloatArray, t: float) -> Tuple[FloatArray, sp.csr_matrix]:
        states = np.asarray(states, dtype=float).reshape(self.n_components, 3)
        residual = np.zeros(self.n_residual, dtype=float)
        jac = sp.lil_matrix((self.n_residual, self.n_dof), dtype=float)
        row = 0
        for constraint in self.constraints:
            r, blocks = constraint.residual_and_blocks(states, t)
            r = np.asarray(r, dtype=float).reshape(-1)
            dim = len(r)
            residual[row : row + dim] = r
            for component, block in blocks.items():
                block = np.asarray(block, dtype=float).reshape(dim, 3)
                col = 3 * component
                jac[row : row + dim, col : col + 3] = block
            row += dim
        return residual, jac.tocsr()


# -----------------------------------------------------------------------------
# Sparse forward kinematics
# -----------------------------------------------------------------------------
@dataclass
class NewtonSolveInfo:
    converged: bool
    iterations: int
    final_energy: float
    residual_norm: float
    energy_history: List[float]
    damping_history: List[float]
    step_norm_history: List[float]


@dataclass
class SimulationResult:
    times: FloatArray
    states: FloatArray
    converged: NDArray[np.bool_]
    residual_norms: FloatArray
    iterations: NDArray[np.int_]
    solve_info: List[NewtonSolveInfo]


@dataclass
class ForwardKinematicsSolver:
    system: ConstraintSystem
    tolerance: float = 1e-10
    max_iterations: int = 60
    initial_damping: float = 1e-8
    maximum_damping: float = 1e8
    armijo: float = 1e-4
    minimum_step: float = 1e-8

    def solve(self, t: float, initial_states: FloatArray) -> Tuple[FloatArray, NewtonSolveInfo]:
        x = np.asarray(initial_states, dtype=float).reshape(-1).copy()
        damping = float(self.initial_damping)
        energies: List[float] = []
        dampings: List[float] = []
        step_norms: List[float] = []

        for iteration in range(1, self.max_iterations + 1):
            states = x.reshape(self.system.n_components, 3)
            residual, jac = self.system.residual_jacobian(states, t)
            energy = 0.5 * float(residual @ residual)
            energies.append(energy)
            dampings.append(damping)

            if not np.isfinite(energy):
                break
            if np.linalg.norm(residual, ord=np.inf) < self.tolerance:
                return states, NewtonSolveInfo(
                    True, iteration - 1, energy, float(np.linalg.norm(residual)),
                    energies, dampings, step_norms,
                )

            gradient = np.asarray(jac.T @ residual).reshape(-1)
            hessian = (jac.T @ jac).tocsc()
            hessian = hessian + damping * sp.eye(hessian.shape[0], format="csc")

            try:
                step = np.asarray(spsolve(hessian, -gradient), dtype=float)
            except Exception:
                damping = min(damping * 100.0, self.maximum_damping)
                continue

            if not np.all(np.isfinite(step)):
                damping = min(damping * 100.0, self.maximum_damping)
                continue

            step_norm = float(np.linalg.norm(step))
            step_norms.append(step_norm)
            if step_norm < self.tolerance:
                return states, NewtonSolveInfo(
                    np.linalg.norm(residual) < 100 * self.tolerance,
                    iteration, energy, float(np.linalg.norm(residual)),
                    energies, dampings, step_norms,
                )

            slope = float(gradient @ step)
            accepted = False
            alpha = 1.0
            while alpha >= self.minimum_step:
                trial = x + alpha * step
                # Keep angles numerically compact without changing the pose.
                trial.reshape(self.system.n_components, 3)[:, 0] = wrap_angle(
                    trial.reshape(self.system.n_components, 3)[:, 0]
                )
                trial_states = trial.reshape(self.system.n_components, 3)
                trial_residual, _ = self.system.residual_jacobian(trial_states, t)
                trial_energy = 0.5 * float(trial_residual @ trial_residual)
                if np.isfinite(trial_energy) and trial_energy <= energy + self.armijo * alpha * slope:
                    x = trial
                    accepted = True
                    damping = max(damping * 0.3, 1e-14)
                    break
                alpha *= 0.5

            if not accepted:
                damping = min(damping * 10.0, self.maximum_damping)
                if damping >= self.maximum_damping:
                    break

        states = x.reshape(self.system.n_components, 3)
        residual, _ = self.system.residual_jacobian(states, t)
        energy = 0.5 * float(residual @ residual)
        return states, NewtonSolveInfo(
            False, len(energies), energy, float(np.linalg.norm(residual)),
            energies, dampings, step_norms,
        )

    def simulate(self, times: ArrayLike, initial_states: FloatArray) -> SimulationResult:
        times = np.asarray(times, dtype=float)
        trajectory = np.zeros((len(times), self.system.n_components, 3), dtype=float)
        converged = np.zeros(len(times), dtype=bool)
        residual_norms = np.zeros(len(times), dtype=float)
        iterations = np.zeros(len(times), dtype=int)
        info: List[NewtonSolveInfo] = []

        guess = np.asarray(initial_states, dtype=float).reshape(self.system.n_components, 3).copy()
        for i, t in enumerate(times):
            solved, solve_info = self.solve(float(t), guess)
            trajectory[i] = solved
            converged[i] = solve_info.converged
            residual_norms[i] = solve_info.residual_norm
            iterations[i] = solve_info.iterations
            info.append(solve_info)
            guess = solved

        return SimulationResult(times, trajectory, converged, residual_norms, iterations, info)


# -----------------------------------------------------------------------------
# Exact four-bar target generator for the included demonstration
# -----------------------------------------------------------------------------
def _circle_intersections(c0: FloatArray, r0: float, c1: FloatArray, r1: float) -> Tuple[FloatArray, FloatArray]:
    delta = c1 - c0
    d = float(np.linalg.norm(delta))
    if d < 1e-12 or d > r0 + r1 + 1e-10 or d < abs(r0 - r1) - 1e-10:
        raise ValueError("The selected four-bar dimensions do not close at this phase.")
    a = (r0**2 - r1**2 + d**2) / (2.0 * d)
    h2 = max(r0**2 - a**2, 0.0)
    h = math.sqrt(h2)
    unit = delta / d
    base = c0 + a * unit
    perp = np.array([-unit[1], unit[0]])
    return base + h * perp, base - h * perp


def generate_fourbar_target(
    times: ArrayLike,
    ground_a: ArrayLike = (0.0, 0.0),
    ground_b: ArrayLike = (3.0, 0.0),
    crank_radius: float = 1.0,
    rocker_radius: float = 2.2,
    coupler_length: float = 2.5,
    phase_offset: float = 0.0,
    initial_branch: str = "upper",
) -> Tuple[List[RigidComponent], FloatArray, Dict[str, Any]]:
    """Generate exact target states from a known four-bar linkage.

    Body 0 is the crank, body 1 is the rocker. The hidden coupler connects
    local [crank_radius,0] to local [rocker_radius,0]. The topology search is
    then asked to rediscover those points from motion alone.
    """
    times = np.asarray(times, dtype=float)
    oa = np.asarray(ground_a, dtype=float)
    ob = np.asarray(ground_b, dtype=float)
    states = np.zeros((len(times), 2, 3), dtype=float)
    previous: Optional[FloatArray] = None

    for i, t in enumerate(times):
        theta_a = float(t + phase_offset)
        pa = oa + rotation(theta_a) @ np.array([crank_radius, 0.0])
        p_plus, p_minus = _circle_intersections(pa, coupler_length, ob, rocker_radius)
        if previous is None:
            chosen = p_plus if initial_branch.lower() == "upper" else p_minus
        else:
            chosen = p_plus if np.linalg.norm(p_plus - previous) <= np.linalg.norm(p_minus - previous) else p_minus
        previous = chosen
        theta_b = math.atan2(chosen[1] - ob[1], chosen[0] - ob[0])
        states[i, 0] = [theta_a, oa[0], oa[1]]
        states[i, 1] = [theta_b, ob[0], ob[1]]

    crank_outline = np.array([[0.0, -0.08], [crank_radius, -0.08], [crank_radius, 0.08], [0.0, 0.08]])
    rocker_outline = np.array([[0.0, -0.10], [rocker_radius, -0.10], [rocker_radius, 0.10], [0.0, 0.10]])
    components = [
        RigidComponent(
            "driver_crank", crank_outline,
            (np.array([0.0, 0.0]), np.array([crank_radius, 0.0])),
            {"tip": np.array([crank_radius, 0.0])},
        ),
        RigidComponent(
            "driven_rocker", rocker_outline,
            (np.array([0.0, 0.0]), np.array([rocker_radius, 0.0])),
            {"tip": np.array([rocker_radius, 0.0]), "hand": np.array([0.85 * rocker_radius, 0.18])},
        ),
    ]
    truth = {
        "ground_a": oa,
        "ground_b": ob,
        "q_a": np.array([crank_radius, 0.0]),
        "q_b": np.array([rocker_radius, 0.0]),
        "length": float(coupler_length),
    }
    return components, states, truth


# -----------------------------------------------------------------------------
# Analytic topology objective
# -----------------------------------------------------------------------------
@dataclass
class TopologyEvaluation:
    total: float
    distance_variance: float
    area_energy: float
    size_energy: float
    gradient: FloatArray
    hessian: FloatArray
    squared_distances: FloatArray
    areas: FloatArray


@dataclass
class TopologyCandidate:
    q_a: FloatArray
    q_b: FloatArray
    length: float
    objective: float
    distance_variance: float
    distance_cv: float
    minimum_moment_arm: float
    percentile05_moment_arm: float
    mean_moment_arm: float
    area_energy: float
    converged: bool
    iterations: int
    start: FloatArray
    history: List[float]

    @property
    def parameters(self) -> FloatArray:
        return np.concatenate([self.q_a, self.q_b])

    def as_dict(self) -> Dict[str, Any]:
        return {
            "q_a": self.q_a.tolist(),
            "q_b": self.q_b.tolist(),
            "length": self.length,
            "objective": self.objective,
            "distance_variance": self.distance_variance,
            "distance_cv": self.distance_cv,
            "minimum_moment_arm": self.minimum_moment_arm,
            "percentile05_moment_arm": self.percentile05_moment_arm,
            "mean_moment_arm": self.mean_moment_arm,
            "area_energy": self.area_energy,
            "converged": self.converged,
            "iterations": self.iterations,
        }


@dataclass
class TopologyObjective:
    states_a: FloatArray
    states_b: FloatArray
    motor_positions: FloatArray
    gamma: float = 0.1
    area_epsilon: float = 1e-12
    area_mode: str = "barrier"
    area_reference: float = 1e-6
    size_regularization: float = 1e-6
    characteristic_length: Optional[float] = None

    def __post_init__(self):
        self.states_a = np.asarray(self.states_a, dtype=float)
        self.states_b = np.asarray(self.states_b, dtype=float)
        self.motor_positions = np.asarray(self.motor_positions, dtype=float)
        if not (len(self.states_a) == len(self.states_b) == len(self.motor_positions)):
            raise ValueError("states_a, states_b, and motor_positions must have equal sample counts.")
        if self.characteristic_length is None:
            cloud = np.vstack([self.states_a[:, 1:3], self.states_b[:, 1:3], self.motor_positions])
            span = np.ptp(cloud, axis=0)
            self.characteristic_length = max(float(np.linalg.norm(span)), 1.0)
        self.characteristic_length = float(self.characteristic_length)

        self._pa: List[FloatArray] = []
        self._pb: List[FloatArray] = []
        self._dmat: List[FloatArray] = []
        self._dc: List[FloatArray] = []
        self._area_a: List[FloatArray] = []
        self._area_b: List[FloatArray] = []
        self._area_ca: List[FloatArray] = []
        self._area_cb: List[FloatArray] = []
        self._area_hessian: List[FloatArray] = []
        k = np.array([[0.0, 1.0], [-1.0, 0.0]], dtype=float)

        for sa, sb, xm in zip(self.states_a, self.states_b, self.motor_positions):
            ra, rb = rotation(sa[0]), rotation(sb[0])
            pa = np.zeros((2, 4), dtype=float)
            pb = np.zeros((2, 4), dtype=float)
            pa[:, 0:2] = ra
            pb[:, 2:4] = rb
            dmat = pa - pb
            dc = sa[1:3] - sb[1:3]

            # area = 0.5 * (xm - xb)^T K (xa - xb)
            aa = -pb
            ab = pa - pb
            ca = xm - sb[1:3]
            cb = sa[1:3] - sb[1:3]
            hess_h = 0.5 * (aa.T @ k @ ab + ab.T @ k.T @ aa)

            self._pa.append(pa)
            self._pb.append(pb)
            self._dmat.append(dmat)
            self._dc.append(dc)
            self._area_a.append(aa)
            self._area_b.append(ab)
            self._area_ca.append(ca)
            self._area_cb.append(cb)
            self._area_hessian.append(hess_h)
        self._k = k

    def world_points(self, z: ArrayLike) -> Tuple[FloatArray, FloatArray]:
        z = np.asarray(z, dtype=float).reshape(4)
        xa, xb = [], []
        for pa, pb, sa, sb in zip(self._pa, self._pb, self.states_a, self.states_b):
            xa.append(pa @ z + sa[1:3])
            xb.append(pb @ z + sb[1:3])
        return np.asarray(xa), np.asarray(xb)

    def evaluate(self, z: ArrayLike, derivatives: bool = True) -> TopologyEvaluation:
        z = np.asarray(z, dtype=float).reshape(4)
        n = len(self.states_a)
        length_scale = self.characteristic_length
        inv_l2 = 1.0 / (length_scale**2)

        g = np.zeros(n, dtype=float)
        p = np.zeros((n, 4), dtype=float)
        g_hess = np.zeros((n, 4, 4), dtype=float)
        areas = np.zeros(n, dtype=float)
        area_grad = np.zeros((n, 4), dtype=float)
        area_hess = np.zeros((n, 4, 4), dtype=float)

        for i, (dmat, dc, aa, ab, ca, cb, hh) in enumerate(
            zip(
                self._dmat, self._dc, self._area_a, self._area_b,
                self._area_ca, self._area_cb, self._area_hessian,
            )
        ):
            d = dmat @ z + dc
            g_raw = float(d @ d)
            g[i] = g_raw * inv_l2
            p[i] = (2.0 * dmat.T @ d) * inv_l2
            g_hess[i] = (2.0 * dmat.T @ dmat) * inv_l2

            avec = aa @ z + ca
            bvec = ab @ z + cb
            h_raw = 0.5 * float(avec @ self._k @ bvec)
            gh_raw = 0.5 * (aa.T @ self._k @ bvec + ab.T @ self._k.T @ avec)
            areas[i] = h_raw
            area_grad[i] = gh_raw * inv_l2
            area_hess[i] = hh * inv_l2

        mean_g = float(np.mean(g))
        e = g - mean_g
        p_bar = np.mean(p, axis=0)
        g_hess_bar = np.mean(g_hess, axis=0)
        distance_variance = float(np.mean(e**2))
        grad_delta = 2.0 * np.mean(e[:, None] * (p - p_bar), axis=0)
        hess_delta = 2.0 * np.mean(
            np.einsum("ni,nj->nij", p - p_bar, p - p_bar)
            + e[:, None, None] * (g_hess - g_hess_bar),
            axis=0,
        )

        area_normalized = areas * inv_l2
        s_area = float(np.mean(area_normalized**2) + self.area_epsilon)
        grad_s = 2.0 * np.mean(area_normalized[:, None] * area_grad, axis=0)
        hess_s = 2.0 * np.mean(
            np.einsum("ni,nj->nij", area_grad, area_grad)
            + area_normalized[:, None, None] * area_hess,
            axis=0,
        )
        if self.area_mode == "paper":
            # Literal expression from the specification. It is unbounded below
            # as anchor coordinates grow, so coordinate bounds are mandatory.
            area_energy = -math.log(s_area)
            grad_area = -grad_s / s_area
            hess_area = np.outer(grad_s, grad_s) / (s_area**2) - hess_s / s_area
        elif self.area_mode == "barrier":
            # Same singular behavior near zero area, but asymptotes to zero for
            # already-safe moment arms instead of rewarding enormous geometry.
            ref = max(float(self.area_reference), self.area_epsilon)
            area_energy = math.log(s_area + ref) - math.log(s_area)
            coefficient = 1.0 / (s_area + ref) - 1.0 / s_area
            coefficient_prime = -1.0 / (s_area + ref) ** 2 + 1.0 / s_area**2
            grad_area = coefficient * grad_s
            hess_area = coefficient * hess_s + coefficient_prime * np.outer(grad_s, grad_s)
        else:
            raise ValueError("area_mode must be 'barrier' or 'paper'.")

        size_energy = self.size_regularization * float(z @ z) / (length_scale**2)
        grad_size = 2.0 * self.size_regularization * z / (length_scale**2)
        hess_size = 2.0 * self.size_regularization * np.eye(4) / (length_scale**2)

        total = distance_variance + self.gamma * area_energy + size_energy
        gradient = grad_delta + self.gamma * grad_area + grad_size
        hessian = hess_delta + self.gamma * hess_area + hess_size
        hessian = 0.5 * (hessian + hessian.T)

        if not derivatives:
            gradient = np.zeros(4)
            hessian = np.zeros((4, 4))

        return TopologyEvaluation(
            total, distance_variance, area_energy, size_energy,
            gradient, hessian, g * length_scale**2, areas,
        )

    def candidate_metrics(self, z: ArrayLike) -> Dict[str, float]:
        ev = self.evaluate(z, derivatives=False)
        distances = np.sqrt(np.maximum(ev.squared_distances, 0.0))
        link_length = float(np.mean(distances))
        cv = float(np.std(distances) / max(link_length, 1e-12))
        moment_arm = 2.0 * np.abs(ev.areas) / np.maximum(distances, 1e-12)
        return {
            "length": link_length,
            "distance_cv": cv,
            "minimum_moment_arm": float(np.min(moment_arm)),
            "percentile05_moment_arm": float(np.percentile(moment_arm, 5.0)),
            "mean_moment_arm": float(np.mean(moment_arm)),
        }


@dataclass
class LocalNewtonResult:
    x: FloatArray
    converged: bool
    iterations: int
    history: List[float]
    gradient_norms: List[float]
    step_norms: List[float]


def modified_newton_minimize(
    objective: TopologyObjective,
    x0: ArrayLike,
    bounds: Tuple[ArrayLike, ArrayLike],
    max_iterations: int = 100,
    gradient_tolerance: float = 1e-9,
    armijo: float = 1e-4,
) -> LocalNewtonResult:
    """Newton's method with Hessian eigenvalue shifting and backtracking."""
    lower = np.asarray(bounds[0], dtype=float).reshape(4)
    upper = np.asarray(bounds[1], dtype=float).reshape(4)
    x = np.clip(np.asarray(x0, dtype=float).reshape(4), lower, upper)
    history: List[float] = []
    grad_history: List[float] = []
    step_history: List[float] = []

    for iteration in range(1, max_iterations + 1):
        ev = objective.evaluate(x)
        f, g, h = ev.total, ev.gradient, ev.hessian
        history.append(float(f))
        gnorm = float(np.linalg.norm(g))
        grad_history.append(gnorm)
        if not np.isfinite(f) or not np.all(np.isfinite(g)) or not np.all(np.isfinite(h)):
            break
        if gnorm < gradient_tolerance:
            return LocalNewtonResult(x, True, iteration - 1, history, grad_history, step_history)

        eig_min = float(np.min(np.linalg.eigvalsh(h)))
        shift = max(0.0, 1e-8 - eig_min)
        h_mod = sp.csc_matrix(h + shift * np.eye(4))
        try:
            step = np.asarray(spsolve(h_mod, -g), dtype=float)
        except Exception:
            step = -g / max(gnorm, 1e-12)
        if not np.all(np.isfinite(step)) or float(g @ step) >= 0.0:
            step = -g / max(gnorm, 1e-12)

        step_history.append(float(np.linalg.norm(step)))
        slope = float(g @ step)
        alpha = 1.0
        accepted = False
        while alpha > 1e-10:
            trial = np.clip(x + alpha * step, lower, upper)
            f_trial = objective.evaluate(trial, derivatives=False).total
            if np.isfinite(f_trial) and f_trial <= f + armijo * alpha * slope:
                x = trial
                accepted = True
                break
            alpha *= 0.5
        if not accepted:
            break
        if np.linalg.norm(alpha * step) < 1e-10:
            return LocalNewtonResult(x, gnorm < 1e-6, iteration, history, grad_history, step_history)

    return LocalNewtonResult(x, False, len(history), history, grad_history, step_history)


def optimize_linkage_points(
    objective: TopologyObjective,
    n_starts: int = 80,
    n_candidates: int = 8,
    coordinate_bound: Optional[float] = None,
    seed: int = 7,
    minimum_length_fraction: float = 0.05,
    minimum_moment_arm_fraction: float = 0.01,
    uniqueness_fraction: float = 0.03,
    max_iterations: int = 100,
) -> List[TopologyCandidate]:
    """Multi-start analytic Newton search for distinct valid link placements."""
    rng = np.random.default_rng(seed)
    scale = objective.characteristic_length
    bound = float(coordinate_bound or 1.25 * scale)
    lower = -bound * np.ones(4)
    upper = bound * np.ones(4)
    raw: List[TopologyCandidate] = []

    starts = [rng.uniform(lower, upper) for _ in range(n_starts)]
    # Include low-discrepancy-looking starts near body origins and along axes.
    starts.extend(
        [
            np.zeros(4),
            np.array([0.25 * scale, 0.0, 0.25 * scale, 0.0]),
            np.array([0.5 * scale, 0.0, -0.5 * scale, 0.0]),
            np.array([0.0, 0.25 * scale, 0.0, -0.25 * scale]),
        ]
    )

    for start in starts:
        result = modified_newton_minimize(
            objective, start, (lower, upper), max_iterations=max_iterations
        )
        z = result.x
        ev = objective.evaluate(z)
        metrics = objective.candidate_metrics(z)
        if not np.isfinite(ev.total):
            continue
        if metrics["length"] < minimum_length_fraction * scale:
            continue
        if metrics["percentile05_moment_arm"] < minimum_moment_arm_fraction * scale:
            continue
        raw.append(
            TopologyCandidate(
                q_a=z[:2].copy(), q_b=z[2:].copy(),
                length=metrics["length"], objective=ev.total,
                distance_variance=ev.distance_variance,
                distance_cv=metrics["distance_cv"],
                minimum_moment_arm=metrics["minimum_moment_arm"],
                percentile05_moment_arm=metrics["percentile05_moment_arm"],
                mean_moment_arm=metrics["mean_moment_arm"],
                area_energy=ev.area_energy,
                converged=result.converged, iterations=result.iterations,
                start=np.asarray(start).copy(), history=result.history,
            )
        )

    # Prefer low variance, then strong low-percentile moment arm.
    raw.sort(key=lambda c: (c.objective, c.distance_cv, -c.percentile05_moment_arm))
    selected: List[TopologyCandidate] = []
    threshold = uniqueness_fraction * scale
    for candidate in raw:
        if all(np.linalg.norm(candidate.parameters - other.parameters) > threshold for other in selected):
            selected.append(candidate)
            if len(selected) >= n_candidates:
                break
    return selected


# -----------------------------------------------------------------------------
# Mechanism assembly and candidate simulation
# -----------------------------------------------------------------------------
def build_two_body_linkage_system(
    components: Sequence[RigidComponent],
    ground_a: ArrayLike,
    ground_b: ArrayLike,
    q_a: ArrayLike,
    q_b: ArrayLike,
    link_length: float,
    motor_angle_function: Callable[[float], float] = lambda t: t,
) -> ConstraintSystem:
    constraints: List[Constraint] = [
        GroundPointConstraint(0, np.zeros(2), np.asarray(ground_a, dtype=float)),
        GroundPointConstraint(1, np.zeros(2), np.asarray(ground_b, dtype=float)),
        MotorAngleConstraint(0, motor_angle_function),
        DistanceConstraint(0, np.asarray(q_a, dtype=float), 1, np.asarray(q_b, dtype=float), float(link_length)),
    ]
    return ConstraintSystem(components, constraints)


def simulate_candidate(
    components: Sequence[RigidComponent],
    target_states: FloatArray,
    times: ArrayLike,
    candidate: TopologyCandidate,
    ground_a: ArrayLike,
    ground_b: ArrayLike,
    solver_kwargs: Optional[Dict[str, Any]] = None,
) -> SimulationResult:
    system = build_two_body_linkage_system(
        components, ground_a, ground_b, candidate.q_a, candidate.q_b,
        candidate.length, motor_angle_function=lambda t: t,
    )
    solver = ForwardKinematicsSolver(system, **(solver_kwargs or {}))
    return solver.simulate(times, target_states[0])


# -----------------------------------------------------------------------------
# Smallest singular value and global objective
# -----------------------------------------------------------------------------
def smallest_singular_value(jacobian: sp.spmatrix, use_sparse: bool = True) -> float:
    """Return sigma_min(J), using sparse partial SVD first and dense fallback."""
    j = jacobian.tocsr()
    m, n = j.shape
    if min(m, n) == 0:
        return 0.0
    if min(m, n) == 1:
        return float(np.linalg.norm(j.toarray()))
    if use_sparse:
        try:
            values = svds(j, k=1, which="SM", return_singular_vectors=False)
            sigma = float(np.min(np.abs(values)))
            if np.isfinite(sigma):
                return sigma
        except Exception:
            pass
    values = np.linalg.svd(j.toarray(), compute_uv=False)
    return float(np.min(values))


@dataclass
class GlobalWeights:
    marker: float = 1.0
    state: float = 0.2
    joint: float = 0.05
    singular: float = 1e-4
    failure: float = 1e5


@dataclass
class GlobalCostBreakdown:
    total: float
    marker: float
    state: float
    joint: float
    singular: float
    failure: float
    link_length: float
    minimum_sigma: float
    simulation: Optional[SimulationResult] = None


@dataclass
class GlobalLinkageObjective:
    components: Sequence[RigidComponent]
    target_states: FloatArray
    times: FloatArray
    ground_a: FloatArray
    ground_b: FloatArray
    marker_specs: Sequence[Tuple[int, FloatArray]]
    bone_segments: Sequence[Tuple[FloatArray, FloatArray]]
    weights: GlobalWeights = field(default_factory=GlobalWeights)
    singular_epsilon: float = 1e-3
    singular_alpha: float = 2.0
    characteristic_length: float = 1.0
    time_stride: int = 1
    solver_kwargs: Dict[str, Any] = field(default_factory=lambda: {"max_iterations": 40, "tolerance": 1e-9})
    cache: Dict[Tuple[float, ...], GlobalCostBreakdown] = field(default_factory=dict)

    def __post_init__(self):
        self.target_states = np.asarray(self.target_states, dtype=float)
        self.times = np.asarray(self.times, dtype=float)
        self.ground_a = np.asarray(self.ground_a, dtype=float)
        self.ground_b = np.asarray(self.ground_b, dtype=float)
        if self.characteristic_length <= 0:
            self.characteristic_length = 1.0

    def _target_marker_trajectory(self, spec: Tuple[int, FloatArray], indices: FloatArray) -> FloatArray:
        body, q = spec
        return np.asarray([world_point(self.target_states[int(i), body], q) for i in indices])

    def evaluate(self, parameters: ArrayLike, keep_simulation: bool = False) -> GlobalCostBreakdown:
        z = np.asarray(parameters, dtype=float).reshape(4)
        key = tuple(np.round(z, 10))
        if key in self.cache and not keep_simulation:
            return self.cache[key]

        indices = np.arange(0, len(self.times), max(1, int(self.time_stride)), dtype=int)
        times = self.times[indices]
        target = self.target_states[indices]

        q_a, q_b = z[:2], z[2:]
        distances = np.array([
            np.linalg.norm(world_point(sa, q_a) - world_point(sb, q_b))
            for sa, sb in zip(target[:, 0], target[:, 1])
        ])
        link_length = float(np.mean(distances))
        if not np.isfinite(link_length) or link_length < 1e-6:
            return GlobalCostBreakdown(self.weights.failure, 0, 0, 0, 0, self.weights.failure, link_length, 0, None)

        system = build_two_body_linkage_system(
            self.components, self.ground_a, self.ground_b,
            q_a, q_b, link_length, lambda t: t,
        )
        solver = ForwardKinematicsSolver(system, **self.solver_kwargs)
        simulation = solver.simulate(times, target[0])
        failure_fraction = 1.0 - float(np.mean(simulation.converged))
        residual_failure = float(np.mean(np.minimum(simulation.residual_norms, 1e3)))
        failure_energy = failure_fraction + residual_failure

        scale2 = self.characteristic_length**2
        marker_energy = 0.0
        for spec in self.marker_specs:
            body, q = spec
            actual = np.asarray([world_point(s[body], q) for s in simulation.states])
            desired = np.asarray([world_point(s[body], q) for s in target])
            marker_energy += float(np.mean(np.sum((actual - desired) ** 2, axis=1)) / scale2)
        marker_energy /= max(len(self.marker_specs), 1)

        angle_error = wrap_angle(simulation.states[:, :, 0] - target[:, :, 0])
        translation_error = simulation.states[:, :, 1:3] - target[:, :, 1:3]
        state_energy = float(
            np.mean(angle_error**2) + np.mean(np.sum(translation_error**2, axis=2)) / scale2
        )

        joint_energy = (
            point_to_segment_distance(q_a, *self.bone_segments[0]) ** 2
            + point_to_segment_distance(q_b, *self.bone_segments[1]) ** 2
        ) / scale2

        sigma_values = []
        singular_terms = []
        for t, states in zip(times, simulation.states):
            _, jac = system.residual_jacobian(states, float(t))
            sigma = smallest_singular_value(jac, use_sparse=True)
            sigma_values.append(sigma)
            singular_terms.append((sigma + self.singular_epsilon) ** (-self.singular_alpha))
        singular_energy = float(np.mean(singular_terms))
        minimum_sigma = float(np.min(sigma_values))

        total = (
            self.weights.marker * marker_energy
            + self.weights.state * state_energy
            + self.weights.joint * joint_energy
            + self.weights.singular * singular_energy
            + self.weights.failure * failure_energy
        )
        breakdown = GlobalCostBreakdown(
            total, marker_energy, state_energy, joint_energy, singular_energy,
            failure_energy, link_length, minimum_sigma,
            simulation if keep_simulation else None,
        )
        if not keep_simulation:
            self.cache[key] = breakdown
        return breakdown

    def __call__(self, parameters: ArrayLike) -> float:
        try:
            result = self.evaluate(parameters, keep_simulation=False)
            return finite_or(result.total, self.weights.failure * 10.0)
        except Exception:
            return self.weights.failure * 10.0


@dataclass
class CMAResult:
    best_parameters: FloatArray
    best_cost: float
    cost_history: List[float]
    sigma_history: List[float]
    evaluations: int
    stop_reasons: Dict[str, Any]
    raw_result: Any


def run_cma_es(
    objective: GlobalLinkageObjective,
    initial_parameters: ArrayLike,
    sigma0: float = 0.15,
    bounds: Optional[Tuple[ArrayLike, ArrayLike]] = None,
    max_iterations: int = 80,
    population_size: Optional[int] = None,
    seed: int = 11,
    verbose: bool = True,
) -> CMAResult:
    """Run pycma using ask-and-tell so notebook plots can use clean histories."""
    try:
        import cma
    except ImportError as exc:
        raise ImportError("Install pycma with: %pip install cma") from exc

    x0 = np.asarray(initial_parameters, dtype=float).reshape(4)
    options: Dict[str, Any] = {
        "seed": seed,
        "maxiter": int(max_iterations),
        "verb_disp": 1 if verbose else 0,
        "verb_log": 0,
    }
    if population_size is not None:
        options["popsize"] = int(population_size)
    if bounds is not None:
        options["bounds"] = [np.asarray(bounds[0]).tolist(), np.asarray(bounds[1]).tolist()]

    es = cma.CMAEvolutionStrategy(x0.tolist(), float(sigma0), options)
    best_history: List[float] = []
    sigma_history: List[float] = []
    while not es.stop():
        solutions = es.ask()
        values = [objective(x) for x in solutions]
        es.tell(solutions, values)
        if verbose:
            es.disp()
        best_history.append(float(es.best.f))
        sigma_history.append(float(es.sigma))
    result = es.result
    return CMAResult(
        best_parameters=np.asarray(result.xbest, dtype=float),
        best_cost=float(result.fbest),
        cost_history=best_history,
        sigma_history=sigma_history,
        evaluations=int(result.evaluations),
        stop_reasons=dict(es.stop()),
        raw_result=result,
    )


# -----------------------------------------------------------------------------
# Plotting and Jupyter interaction
# -----------------------------------------------------------------------------
def _draw_component(ax, component: RigidComponent, state: FloatArray, alpha: float = 0.9):
    outline = component.transformed_outline(state)
    patch = Polygon(outline, closed=True, fill=False, linewidth=2.0, alpha=alpha)
    ax.add_patch(patch)
    origin = state[1:3]
    ax.plot(origin[0], origin[1], "o", markersize=6)


def draw_mechanism(
    ax,
    components: Sequence[RigidComponent],
    states: FloatArray,
    q_a: Optional[ArrayLike] = None,
    q_b: Optional[ArrayLike] = None,
    motor_position: Optional[ArrayLike] = None,
    target_states: Optional[FloatArray] = None,
    title: Optional[str] = None,
):
    ax.clear()
    if target_states is not None:
        for component, state in zip(components, target_states):
            outline = component.transformed_outline(state)
            ax.plot(*np.vstack([outline, outline[0]]).T, linestyle=":", linewidth=1.2, alpha=0.55)
    for component, state in zip(components, states):
        _draw_component(ax, component, state)
    if q_a is not None and q_b is not None:
        pa = world_point(states[0], q_a)
        pb = world_point(states[1], q_b)
        ax.plot([pa[0], pb[0]], [pa[1], pb[1]], linewidth=3.0)
        ax.plot([pa[0], pb[0]], [pa[1], pb[1]], "o", markersize=7)
        if motor_position is not None:
            xm = np.asarray(motor_position, dtype=float)
            ax.fill([pa[0], pb[0], xm[0]], [pa[1], pb[1], xm[1]], alpha=0.12)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    if title:
        ax.set_title(title)


def plot_target_motion(
    components: Sequence[RigidComponent],
    target_states: FloatArray,
    sample_count: int = 12,
    marker: Tuple[int, ArrayLike] = (1, np.array([2.2, 0.0])),
):
    fig, ax = plt.subplots(figsize=(10, 7))
    indices = np.linspace(0, len(target_states) - 1, sample_count, dtype=int)
    for rank, idx in enumerate(indices):
        alpha = 0.15 + 0.75 * rank / max(len(indices) - 1, 1)
        for component, state in zip(components, target_states[idx]):
            outline = component.transformed_outline(state)
            ax.plot(*np.vstack([outline, outline[0]]).T, alpha=alpha)
    body, q = marker
    trajectory = np.asarray([world_point(s[body], q) for s in target_states])
    ax.plot(trajectory[:, 0], trajectory[:, 1], linewidth=2.5, label="target marker path")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    ax.set_title("Input periodic motion and marker trajectory")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend()
    plt.show()
    return fig, ax


def plot_newton_history(candidate: TopologyCandidate):
    fig, ax = plt.subplots(figsize=(9, 5))
    values = np.asarray(candidate.history, dtype=float)
    ax.semilogy(np.arange(len(values)), np.maximum(values - np.min(values) + 1e-16, 1e-16))
    ax.set_xlabel("Newton iteration")
    ax.set_ylabel("shifted objective")
    ax.set_title("Analytic topology optimizer convergence")
    ax.grid(True, which="both", alpha=0.3)
    plt.show()
    return fig, ax


def plot_candidate_metrics(candidates: Sequence[TopologyCandidate]):
    fig, ax = plt.subplots(figsize=(9, 6))
    x = [c.distance_cv for c in candidates]
    y = [c.percentile05_moment_arm for c in candidates]
    sizes = 80 + 180 * np.asarray([c.mean_moment_arm for c in candidates]) / max(max(c.mean_moment_arm for c in candidates), 1e-12)
    ax.scatter(x, y, s=sizes, alpha=0.75)
    for i, (xx, yy) in enumerate(zip(x, y)):
        ax.annotate(str(i), (xx, yy), xytext=(5, 5), textcoords="offset points")
    ax.set_xscale("log")
    ax.set_xlabel("distance coefficient of variation, lower is better")
    ax.set_ylabel("5th-percentile moment arm, higher is better")
    ax.set_title("Candidate linkage quality map")
    ax.grid(True, which="both", alpha=0.3)
    plt.show()
    return fig, ax


def plot_candidate_gallery(
    candidates: Sequence[TopologyCandidate],
    components: Sequence[RigidComponent],
    target_states: FloatArray,
    phase_index: int = 0,
    motor_position: Optional[ArrayLike] = None,
    columns: int = 3,
):
    if not candidates:
        raise ValueError("No candidates to display.")
    rows = int(math.ceil(len(candidates) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(5 * columns, 4.5 * rows), squeeze=False)
    for i, ax in enumerate(axes.flat):
        if i >= len(candidates):
            ax.axis("off")
            continue
        c = candidates[i]
        draw_mechanism(
            ax, components, target_states[phase_index], c.q_a, c.q_b,
            motor_position=motor_position,
            title=f"Candidate {i}\nCV={c.distance_cv:.2e}, p05 arm={c.percentile05_moment_arm:.3f}",
        )
    fig.tight_layout()
    plt.show()
    return fig, axes


def plot_objective_slice(
    objective: TopologyObjective,
    center: ArrayLike,
    parameter_x: int = 0,
    parameter_y: int = 2,
    half_width: Optional[float] = None,
    resolution: int = 120,
):
    center = np.asarray(center, dtype=float).reshape(4)
    width = float(half_width or 0.3 * objective.characteristic_length)
    xs = np.linspace(center[parameter_x] - width, center[parameter_x] + width, resolution)
    ys = np.linspace(center[parameter_y] - width, center[parameter_y] + width, resolution)
    zz = np.zeros((resolution, resolution), dtype=float)
    for iy, y in enumerate(ys):
        for ix, x in enumerate(xs):
            p = center.copy()
            p[parameter_x] = x
            p[parameter_y] = y
            zz[iy, ix] = objective.evaluate(p, derivatives=False).total
    floor = np.nanpercentile(zz, 2)
    ceiling = np.nanpercentile(zz, 95)
    shown = np.clip(zz, floor, ceiling)
    fig, ax = plt.subplots(figsize=(9, 7))
    image = ax.imshow(
        shown, origin="lower", extent=[xs[0], xs[-1], ys[0], ys[-1]],
        aspect="auto",
    )
    ax.plot(center[parameter_x], center[parameter_y], "x", markersize=12, markeredgewidth=3)
    ax.set_xlabel(f"parameter z[{parameter_x}]")
    ax.set_ylabel(f"parameter z[{parameter_y}]")
    ax.set_title("Local slice through topology objective")
    fig.colorbar(image, ax=ax, label="clipped objective")
    plt.show()
    return fig, ax


def plot_link_quality_over_cycle(
    objective: TopologyObjective,
    candidate: TopologyCandidate,
    phases: Optional[ArrayLike] = None,
):
    ev = objective.evaluate(candidate.parameters, derivatives=False)
    distances = np.sqrt(np.maximum(ev.squared_distances, 0.0))
    moment_arm = 2.0 * np.abs(ev.areas) / np.maximum(distances, 1e-12)
    x = np.arange(len(distances)) if phases is None else np.asarray(phases)

    fig1, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(x, distances, linewidth=2.0)
    ax1.axhline(candidate.length, linestyle="--", label=f"mean length = {candidate.length:.5f}")
    ax1.set_xlabel("phase" if phases is not None else "sample index")
    ax1.set_ylabel("anchor distance")
    ax1.set_title("How constant is the proposed rigid-link length?")
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    plt.show()

    fig2, ax2 = plt.subplots(figsize=(10, 5))
    ax2.plot(x, moment_arm, linewidth=2.0)
    ax2.axhline(candidate.percentile05_moment_arm, linestyle="--", label="5th percentile")
    ax2.set_xlabel("phase" if phases is not None else "sample index")
    ax2.set_ylabel("moment arm")
    ax2.set_title("Torque transmission margin across the cycle")
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    plt.show()
    return (fig1, ax1), (fig2, ax2)


def plot_forward_diagnostics(
    simulation: SimulationResult,
    target_states: FloatArray,
    components: Sequence[RigidComponent],
    marker_spec: Tuple[int, ArrayLike] = (1, np.array([2.2, 0.0])),
):
    body, q = marker_spec
    actual_marker = np.asarray([world_point(s[body], q) for s in simulation.states])
    target_marker = np.asarray([world_point(s[body], q) for s in target_states[: len(simulation.states)]])

    fig1, ax1 = plt.subplots(figsize=(8, 7))
    ax1.plot(target_marker[:, 0], target_marker[:, 1], linewidth=3, label="target")
    ax1.plot(actual_marker[:, 0], actual_marker[:, 1], linestyle="--", linewidth=2, label="linkage")
    ax1.set_aspect("equal", adjustable="box")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_title("End-effector trajectory reproduction")
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    plt.show()

    state_error = simulation.states - target_states[: len(simulation.states)]
    state_error[:, :, 0] = wrap_angle(state_error[:, :, 0])
    rms = np.sqrt(np.mean(state_error**2, axis=1))
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    ax2.plot(simulation.times, rms[:, 0], label="angle RMS")
    ax2.plot(simulation.times, rms[:, 1], label="x RMS")
    ax2.plot(simulation.times, rms[:, 2], label="y RMS")
    ax2.set_xlabel("phase")
    ax2.set_ylabel("RMS state error")
    ax2.set_title("Rigid-body state error across the cycle")
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    plt.show()

    fig3, ax3 = plt.subplots(figsize=(10, 5))
    ax3.semilogy(simulation.times, np.maximum(simulation.residual_norms, 1e-16), label="constraint residual")
    ax3.plot(simulation.times, simulation.iterations, label="Newton iterations")
    ax3.set_xlabel("phase")
    ax3.set_title("Forward solver health")
    ax3.grid(True, which="both", alpha=0.3)
    ax3.legend()
    plt.show()
    return (fig1, ax1), (fig2, ax2), (fig3, ax3)


def plot_singularity_profile(system: ConstraintSystem, simulation: SimulationResult):
    sigmas = []
    conditions = []
    for t, states in zip(simulation.times, simulation.states):
        _, jac = system.residual_jacobian(states, float(t))
        dense_s = np.linalg.svd(jac.toarray(), compute_uv=False)
        sigmas.append(float(np.min(dense_s)))
        conditions.append(float(np.max(dense_s) / max(np.min(dense_s), 1e-15)))

    fig1, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(simulation.times, sigmas, linewidth=2)
    ax1.set_xlabel("phase")
    ax1.set_ylabel("smallest singular value")
    ax1.set_title("Constraint-Jacobian singularity margin")
    ax1.grid(True, alpha=0.3)
    plt.show()

    fig2, ax2 = plt.subplots(figsize=(10, 5))
    ax2.semilogy(simulation.times, conditions, linewidth=2)
    ax2.set_xlabel("phase")
    ax2.set_ylabel("condition number")
    ax2.set_title("Constraint-Jacobian conditioning")
    ax2.grid(True, which="both", alpha=0.3)
    plt.show()
    return np.asarray(sigmas), np.asarray(conditions)


def plot_cma_history(result: CMAResult):
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    ax1.semilogy(np.maximum(result.cost_history, 1e-16), linewidth=2)
    ax1.set_xlabel("generation")
    ax1.set_ylabel("best cost")
    ax1.set_title("CMA-ES global refinement")
    ax1.grid(True, which="both", alpha=0.3)
    plt.show()

    fig2, ax2 = plt.subplots(figsize=(10, 5))
    ax2.semilogy(np.maximum(result.sigma_history, 1e-16), linewidth=2)
    ax2.set_xlabel("generation")
    ax2.set_ylabel("CMA step size")
    ax2.set_title("CMA-ES search radius")
    ax2.grid(True, which="both", alpha=0.3)
    plt.show()
    return (fig1, ax1), (fig2, ax2)


def interactive_phase_viewer(
    components: Sequence[RigidComponent],
    states: FloatArray,
    q_a: Optional[ArrayLike] = None,
    q_b: Optional[ArrayLike] = None,
    motor_positions: Optional[FloatArray] = None,
    target_states: Optional[FloatArray] = None,
):
    """Create an ipywidgets phase slider in JupyterLab."""
    try:
        import ipywidgets as widgets
        from IPython.display import display
    except ImportError as exc:
        raise ImportError("Install widgets with: %pip install ipywidgets ipympl") from exc

    slider = widgets.IntSlider(
        value=0, min=0, max=len(states) - 1, step=1,
        description="phase", continuous_update=True,
        layout=widgets.Layout(width="700px"),
    )
    output = widgets.Output()

    def redraw(change=None):
        idx = slider.value
        with output:
            output.clear_output(wait=True)
            fig, ax = plt.subplots(figsize=(9, 7))
            motor = None if motor_positions is None else motor_positions[idx]
            target = None if target_states is None else target_states[idx]
            draw_mechanism(
                ax, components, states[idx], q_a, q_b,
                motor_position=motor, target_states=target,
                title=f"Mechanism phase sample {idx}/{len(states)-1}",
            )
            plt.show()

    slider.observe(redraw, names="value")
    redraw()
    display(widgets.VBox([slider, output]))
    return slider, output


def interactive_candidate_dashboard(
    candidates: Sequence[TopologyCandidate],
    objective: TopologyObjective,
    components: Sequence[RigidComponent],
    states: FloatArray,
    motor_positions: Optional[FloatArray] = None,
):
    try:
        import ipywidgets as widgets
        from IPython.display import display
    except ImportError as exc:
        raise ImportError("Install widgets with: %pip install ipywidgets ipympl") from exc
    if not candidates:
        raise ValueError("No candidates were supplied.")

    candidate_slider = widgets.IntSlider(value=0, min=0, max=len(candidates)-1, description="candidate")
    phase_slider = widgets.IntSlider(value=0, min=0, max=len(states)-1, description="phase")
    output = widgets.Output()

    def redraw(change=None):
        c = candidates[candidate_slider.value]
        idx = phase_slider.value
        ev = objective.evaluate(c.parameters, derivatives=False)
        distances = np.sqrt(np.maximum(ev.squared_distances, 0.0))
        arms = 2.0 * np.abs(ev.areas) / np.maximum(distances, 1e-12)
        with output:
            output.clear_output(wait=True)
            fig, ax = plt.subplots(figsize=(9, 7))
            motor = None if motor_positions is None else motor_positions[idx]
            draw_mechanism(
                ax, components, states[idx], c.q_a, c.q_b, motor,
                title=(
                    f"Candidate {candidate_slider.value} | phase {idx}\n"
                    f"L={c.length:.4f}, CV={c.distance_cv:.2e}, "
                    f"arm={arms[idx]:.4f}"
                ),
            )
            plt.show()
            print(json.dumps(c.as_dict(), indent=2))

    candidate_slider.observe(redraw, names="value")
    phase_slider.observe(redraw, names="value")
    redraw()
    display(widgets.VBox([widgets.HBox([candidate_slider, phase_slider]), output]))
    return candidate_slider, phase_slider, output


def animate_mechanism(
    components: Sequence[RigidComponent],
    states: FloatArray,
    q_a: Optional[ArrayLike] = None,
    q_b: Optional[ArrayLike] = None,
    interval_ms: int = 50,
):
    fig, ax = plt.subplots(figsize=(9, 7))

    def update(frame):
        draw_mechanism(ax, components, states[frame], q_a, q_b, title=f"frame {frame}")
        return ax.patches + ax.lines

    animation = FuncAnimation(fig, update, frames=len(states), interval=interval_ms, blit=False)
    plt.close(fig)
    return animation


# -----------------------------------------------------------------------------
# Organic link shaping with Catmull-Rom centerlines
# -----------------------------------------------------------------------------
def catmull_rom_chain(points: ArrayLike, samples_per_segment: int = 30, alpha: float = 0.5) -> FloatArray:
    """Centripetal Catmull-Rom interpolation through 2D control points."""
    p = np.asarray(points, dtype=float)
    if len(p) < 2:
        raise ValueError("At least two control points are required.")
    # Endpoint duplication gives an interpolating open curve.
    extended = np.vstack([p[0], p, p[-1]])
    curve: List[FloatArray] = []

    def tj(ti: float, pi: FloatArray, pj: FloatArray) -> float:
        return ti + max(float(np.linalg.norm(pj - pi)), 1e-12) ** alpha

    for i in range(len(extended) - 3):
        p0, p1, p2, p3 = extended[i : i + 4]
        t0 = 0.0
        t1 = tj(t0, p0, p1)
        t2 = tj(t1, p1, p2)
        t3 = tj(t2, p2, p3)
        ts = np.linspace(t1, t2, samples_per_segment, endpoint=(i == len(extended)-4))
        for t in ts:
            a1 = (t1 - t) / max(t1 - t0, 1e-12) * p0 + (t - t0) / max(t1 - t0, 1e-12) * p1
            a2 = (t2 - t) / max(t2 - t1, 1e-12) * p1 + (t - t1) / max(t2 - t1, 1e-12) * p2
            a3 = (t3 - t) / max(t3 - t2, 1e-12) * p2 + (t - t2) / max(t3 - t2, 1e-12) * p3
            b1 = (t2 - t) / max(t2 - t0, 1e-12) * a1 + (t - t0) / max(t2 - t0, 1e-12) * a2
            b2 = (t3 - t) / max(t3 - t1, 1e-12) * a2 + (t - t1) / max(t3 - t1, 1e-12) * a3
            c = (t2 - t) / max(t2 - t1, 1e-12) * b1 + (t - t1) / max(t2 - t1, 1e-12) * b2
            curve.append(c)
    return np.asarray(curve)


def fit_organic_link_centerline(
    states_a: FloatArray,
    states_b: FloatArray,
    q_a: ArrayLike,
    q_b: ArrayLike,
    bend_fraction: float = 0.10,
    tangent_fraction: float = 0.18,
    samples_per_segment: int = 35,
) -> Tuple[FloatArray, Dict[str, float]]:
    """Create one rigid, organic-looking centerline in the link's own frame.

    Since the fabricated link is rigid, its shape cannot change with phase.
    Endpoint tangent angles are selected by circular averaging of the adjacent
    body-axis directions expressed in the moving link frame. This minimizes
    mean wrapped angular deviation over the sampled cycle.
    """
    q_a = np.asarray(q_a, dtype=float)
    q_b = np.asarray(q_b, dtype=float)
    rel_a, rel_b, lengths = [], [], []
    for sa, sb in zip(states_a, states_b):
        pa = world_point(sa, q_a)
        pb = world_point(sb, q_b)
        v = pb - pa
        link_angle = math.atan2(v[1], v[0])
        lengths.append(np.linalg.norm(v))
        rel_a.append(float(wrap_angle(sa[0] - link_angle)))
        # Tangent at endpoint B points backward into the link.
        rel_b.append(float(wrap_angle(sb[0] - (link_angle + np.pi))))
    length = float(np.mean(lengths))
    theta0 = circular_mean(rel_a)
    theta1 = circular_mean(rel_b)
    d = tangent_fraction * length
    bend = bend_fraction * length
    start = np.array([0.0, 0.0])
    end = np.array([length, 0.0])
    control = np.array([
        start,
        start + d * np.array([math.cos(theta0), math.sin(theta0)]),
        np.array([0.5 * length, bend]),
        end + d * np.array([math.cos(theta1), math.sin(theta1)]),
        end,
    ])
    curve = catmull_rom_chain(control, samples_per_segment=samples_per_segment)
    metadata = {
        "length": length,
        "start_tangent_angle": theta0,
        "end_tangent_angle": theta1,
        "bend_fraction": bend_fraction,
    }
    return curve, metadata


def plot_link_centerline(centerline: FloatArray, hole_radius: float = 0.08):
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(centerline[:, 0], centerline[:, 1], linewidth=4)
    ax.add_patch(plt.Circle(centerline[0], hole_radius, fill=False, linewidth=2))
    ax.add_patch(plt.Circle(centerline[-1], hole_radius, fill=False, linewidth=2))
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    ax.set_title("Rigid organic linkage centerline and pin holes")
    ax.set_xlabel("link-local x")
    ax.set_ylabel("link-local y")
    plt.show()
    return fig, ax


# -----------------------------------------------------------------------------
# Collision graph and fabrication layers
# -----------------------------------------------------------------------------
def _orientation(a: FloatArray, b: FloatArray, c: FloatArray) -> float:
    return float(np.cross(b - a, c - a))


def segments_intersect(a: ArrayLike, b: ArrayLike, c: ArrayLike, d: ArrayLike, tolerance: float = 1e-10) -> bool:
    a, b, c, d = map(lambda x: np.asarray(x, dtype=float), (a, b, c, d))
    o1, o2 = _orientation(a, b, c), _orientation(a, b, d)
    o3, o4 = _orientation(c, d, a), _orientation(c, d, b)
    return (o1 * o2 < -tolerance) and (o3 * o4 < -tolerance)


def build_intersection_graph(
    link_trajectories: Dict[str, FloatArray],
    ignore_shared_endpoints: bool = True,
    endpoint_tolerance: float = 1e-6,
):
    """Build an undirected graph whose edges mean two moving links intersect."""
    try:
        import networkx as nx
    except ImportError as exc:
        raise ImportError("Install NetworkX with: %pip install networkx") from exc
    names = list(link_trajectories)
    graph = nx.Graph()
    graph.add_nodes_from(names)
    for i, name_i in enumerate(names):
        traj_i = np.asarray(link_trajectories[name_i], dtype=float)
        for name_j in names[i + 1 :]:
            traj_j = np.asarray(link_trajectories[name_j], dtype=float)
            collision = False
            for seg_i, seg_j in zip(traj_i, traj_j):
                a, b = seg_i
                c, d = seg_j
                if ignore_shared_endpoints:
                    shared = min(
                        np.linalg.norm(a-c), np.linalg.norm(a-d),
                        np.linalg.norm(b-c), np.linalg.norm(b-d),
                    ) < endpoint_tolerance
                    if shared:
                        continue
                if segments_intersect(a, b, c, d):
                    collision = True
                    break
            if collision:
                graph.add_edge(name_i, name_j)
    return graph


def assign_depth_layers(intersection_graph, layer_spacing: float = 2.0) -> Dict[str, Dict[str, float]]:
    """Assign Z layers by graph coloring.

    An intersection graph is undirected, so topological sorting alone is not
    mathematically applicable. Coloring is the correct separation problem:
    intersecting links receive different colors/layers. If a design also has
    directed precedence constraints, topological sorting can be applied after
    those directed constraints are supplied.
    """
    try:
        import networkx as nx
    except ImportError as exc:
        raise ImportError("Install NetworkX with: %pip install networkx") from exc
    colors = nx.coloring.greedy_color(intersection_graph, strategy="saturation_largest_first")
    return {
        name: {"layer": int(color), "z_offset": float(color * layer_spacing)}
        for name, color in colors.items()
    }


def plot_intersection_graph(graph, layer_assignment: Optional[Dict[str, Dict[str, float]]] = None):
    try:
        import networkx as nx
    except ImportError as exc:
        raise ImportError("Install NetworkX with: %pip install networkx") from exc
    fig, ax = plt.subplots(figsize=(9, 7))
    pos = nx.spring_layout(graph, seed=3)
    node_values = None
    if layer_assignment:
        node_values = [layer_assignment[n]["layer"] for n in graph.nodes]
    nx.draw_networkx(graph, pos=pos, node_color=node_values, ax=ax, with_labels=True)
    ax.set_title("Full-cycle linkage intersection graph")
    ax.axis("off")
    plt.show()
    return fig, ax


# -----------------------------------------------------------------------------
# STL and JSON export
# -----------------------------------------------------------------------------
def export_link_stl(
    centerline: ArrayLike,
    output_path: str | Path,
    width: float = 0.22,
    thickness: float = 0.12,
    hole_radius: float = 0.055,
    layer_z: float = 0.0,
) -> Path:
    """Buffer a spline centerline, subtract pin holes, extrude, and export STL."""
    try:
        import trimesh
        from shapely.geometry import LineString, Point
    except ImportError as exc:
        raise ImportError("Install fabrication packages with: %pip install trimesh shapely mapbox_earcut") from exc

    centerline = np.asarray(centerline, dtype=float)
    polygon = LineString(centerline).buffer(width / 2.0, cap_style=1, join_style=1)
    polygon = polygon.difference(Point(centerline[0]).buffer(hole_radius))
    polygon = polygon.difference(Point(centerline[-1]).buffer(hole_radius))
    if polygon.is_empty or not polygon.is_valid:
        raise ValueError("Generated 2D linkage polygon is invalid. Reduce bend or hole radius.")
    try:
        mesh = trimesh.creation.extrude_polygon(polygon, height=thickness)
    except Exception as exc:
        raise RuntimeError(
            "Polygon triangulation failed. In Jupyter run: %pip install mapbox_earcut"
        ) from exc
    mesh.apply_translation([0.0, 0.0, layer_z])
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(path)
    return path


def export_design_json(
    output_path: str | Path,
    candidate: TopologyCandidate,
    spline_metadata: Optional[Dict[str, Any]] = None,
    layer_assignment: Optional[Dict[str, Any]] = None,
    thickness: Optional[float] = None,
    width: Optional[float] = None,
) -> Path:
    payload = {
        "linkage": candidate.as_dict(),
        "spline": spline_metadata or {},
        "layers": layer_assignment or {},
        "fabrication": {"thickness": thickness, "width": width},
    }
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


# -----------------------------------------------------------------------------
# Verification helpers
# -----------------------------------------------------------------------------
def check_topology_derivatives(
    objective: TopologyObjective,
    z: ArrayLike,
    step: float = 1e-5,
) -> Dict[str, float]:
    """Compare analytic gradient/Hessian to central finite differences."""
    z = np.asarray(z, dtype=float).reshape(4)
    ev = objective.evaluate(z)
    numerical_gradient = np.zeros(4)
    numerical_hessian = np.zeros((4, 4))
    for i in range(4):
        ei = np.zeros(4); ei[i] = step
        fp = objective.evaluate(z + ei, derivatives=False).total
        fm = objective.evaluate(z - ei, derivatives=False).total
        numerical_gradient[i] = (fp - fm) / (2 * step)
        gp = objective.evaluate(z + ei).gradient
        gm = objective.evaluate(z - ei).gradient
        numerical_hessian[:, i] = (gp - gm) / (2 * step)
    numerical_hessian = 0.5 * (numerical_hessian + numerical_hessian.T)
    return {
        "gradient_relative_error": float(
            np.linalg.norm(ev.gradient - numerical_gradient)
            / max(np.linalg.norm(numerical_gradient), 1e-12)
        ),
        "hessian_relative_error": float(
            np.linalg.norm(ev.hessian - numerical_hessian)
            / max(np.linalg.norm(numerical_hessian), 1e-12)
        ),
    }


def candidate_table(candidates: Sequence[TopologyCandidate]):
    """Return a pandas DataFrame when pandas is available, otherwise records."""
    records = []
    for i, c in enumerate(candidates):
        records.append({
            "candidate": i,
            "u_a": c.q_a[0], "v_a": c.q_a[1],
            "u_b": c.q_b[0], "v_b": c.q_b[1],
            "length": c.length,
            "distance_cv": c.distance_cv,
            "p05_moment_arm": c.percentile05_moment_arm,
            "min_moment_arm": c.minimum_moment_arm,
            "objective": c.objective,
            "converged": c.converged,
        })
    try:
        import pandas as pd
        return pd.DataFrame.from_records(records)
    except ImportError:
        return records


# -----------------------------------------------------------------------------
# End-to-end demonstration
# -----------------------------------------------------------------------------
def run_demo(
    n_samples: int = 121,
    n_starts: int = 80,
    n_candidates: int = 8,
    gamma: float = 0.1,
    seed: int = 7,
    make_plots: bool = True,
) -> Dict[str, Any]:
    times = np.linspace(0.0, 2.0 * np.pi, n_samples, endpoint=False)
    components, target_states, truth = generate_fourbar_target(times)
    motor_positions = np.repeat(truth["ground_b"][None, :], n_samples, axis=0)
    objective = TopologyObjective(
        target_states[:, 0], target_states[:, 1], motor_positions,
        gamma=gamma, characteristic_length=4.0,
    )
    derivative_check = check_topology_derivatives(
        objective, np.array([0.8, 0.15, 2.0, -0.1])
    )
    candidates = optimize_linkage_points(
        objective, n_starts=n_starts, n_candidates=n_candidates, seed=seed,
        coordinate_bound=3.5, minimum_moment_arm_fraction=0.005,
    )
    if not candidates:
        raise RuntimeError("No valid candidates found. Increase n_starts or relax moment-arm threshold.")
    best = candidates[0]
    simulation = simulate_candidate(
        components, target_states, times, best,
        truth["ground_a"], truth["ground_b"],
    )
    system = build_two_body_linkage_system(
        components, truth["ground_a"], truth["ground_b"],
        best.q_a, best.q_b, best.length, lambda t: t,
    )

    if make_plots:
        plot_target_motion(components, target_states, marker=(1, components[1].markers["tip"]))
        plot_candidate_metrics(candidates)
        plot_candidate_gallery(candidates, components, target_states, motor_position=truth["ground_b"])
        plot_link_quality_over_cycle(objective, best, times)
        plot_forward_diagnostics(simulation, target_states, components, marker_spec=(1, components[1].markers["tip"]))
        plot_singularity_profile(system, simulation)

    return {
        "times": times,
        "components": components,
        "target_states": target_states,
        "truth": truth,
        "motor_positions": motor_positions,
        "topology_objective": objective,
        "derivative_check": derivative_check,
        "candidates": candidates,
        "best_candidate": best,
        "simulation": simulation,
        "system": system,
    }


if __name__ == "__main__":
    result = run_demo(n_samples=91, n_starts=40, n_candidates=6, make_plots=True)
    print("Derivative check:", result["derivative_check"])
    print(candidate_table(result["candidates"]))
