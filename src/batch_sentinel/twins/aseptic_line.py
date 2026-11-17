"""Aseptic fill-finish cleanroom differential-pressure and 6-DOF robotics twin."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from ..config import FillFinishParams


@dataclass(frozen=True)
class VialFillRecord:
    """Individual serialized vial dosing and cleanroom telemetry."""

    vial_index: int
    dosed_volume_ml: float
    cleanroom_dp_pa: float
    ik_residual_m: float
    joint_angles_rad: Tuple[float, ...]
    interlock_halted: bool
    accepted: bool


@dataclass
class AsepticLineResult:
    """Full aseptic fill-finish batch result."""

    vials: List[VialFillRecord]
    mean_fill_ml: float
    min_dp_pa: float
    max_ik_error_m: float
    acceptance_rate_pct: float
    airlock_breach_halted: bool
    end_effector_path_xyz: List[Tuple[float, float, float]]


class SixDOFRobotArm:
    """6-DOF Denavit-Hartenberg manipulator for sterile vial pick-and-place.

    Standard DH table ``(a_i, alpha_i, d_i, theta_i)`` in meters and radians,
    modeling a pharmaceutical isolator articulated arm with wrist pitch/roll.
    """

    def __init__(
        self,
        damping: float = 0.025,
        max_iters: int = 80,
        tol_m: float = 1e-4,
    ) -> None:
        self.damping = damping
        self.max_iters = max_iters
        self.tol_m = tol_m
        # (a, alpha, d) for joints 1..6 (meters, radians)
        self.dh_params: Tuple[Tuple[float, float, float], ...] = (
            (0.0, math.pi / 2.0, 0.32),
            (0.42, 0.0, 0.0),
            (0.36, 0.0, 0.0),
            (0.0, math.pi / 2.0, 0.14),
            (0.0, -math.pi / 2.0, 0.10),
            (0.0, 0.0, 0.08),
        )
        self.joint_limits: Tuple[Tuple[float, float], ...] = (
            (-3.14, 3.14),
            (-1.8, 2.2),
            (-2.4, 2.4),
            (-3.14, 3.14),
            (-2.1, 2.1),
            (-3.14, 3.14),
        )

    @staticmethod
    def _dh_matrix(a: float, alpha: float, d: float, theta: float) -> NDArray[np.float64]:
        ct, st = math.cos(theta), math.sin(theta)
        ca, sa = math.cos(alpha), math.sin(alpha)
        return np.array(
            [
                [ct, -st * ca, st * sa, a * ct],
                [st, ct * ca, -ct * sa, a * st],
                [0.0, sa, ca, d],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

    def forward_kinematics(self, q: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute 4x4 homogeneous end-effector pose ``T_06(q)``."""
        t = np.eye(4, dtype=np.float64)
        for idx, (a, alpha, d) in enumerate(self.dh_params):
            t = t @ self._dh_matrix(a, alpha, d, float(q[idx]))
        return t

    def position_jacobian(self, q: NDArray[np.float64], eps: float = 1e-6) -> NDArray[np.float64]:
        """Compute 3x6 translational Jacobian ``J_v(q)`` via central finite differences."""
        jac = np.zeros((3, 6), dtype=np.float64)
        for j in range(6):
            qp = q.copy()
            qm = q.copy()
            qp[j] += eps
            qm[j] -= eps
            pp = self.forward_kinematics(qp)[:3, 3]
            pm = self.forward_kinematics(qm)[:3, 3]
            jac[:, j] = (pp - pm) / (2.0 * eps)
        return jac

    def solve_ik_position(
        self,
        target_xyz: Tuple[float, float, float],
        q_init: Optional[NDArray[np.float64]] = None,
    ) -> Tuple[NDArray[np.float64], float]:
        """Solve 3D target position via Levenberg-Marquardt damped least-squares IK."""
        q = (
            np.array(q_init, dtype=np.float64, copy=True)
            if q_init is not None
            else np.array([0.2, 0.6, -0.5, 0.0, 0.3, 0.0], dtype=np.float64)
        )
        target = np.asarray(target_xyz, dtype=np.float64)
        lam2 = self.damping ** 2
        eye3 = np.eye(3, dtype=np.float64)

        err_norm = float("inf")
        for _ in range(self.max_iters):
            pos = self.forward_kinematics(q)[:3, 3]
            err = target - pos
            err_norm = float(np.linalg.norm(err))
            if err_norm <= self.tol_m:
                break
            jac = self.position_jacobian(q)
            # Damped pseudoinverse: dq = J^T (J J^T + lam^2 I)^-1 err
            jjt = jac @ jac.T + lam2 * eye3
            dq = jac.T @ np.linalg.solve(jjt, err)
            q = q + 0.85 * dq
            for j, (lo, hi) in enumerate(self.joint_limits):
                q[j] = float(np.clip(q[j], lo, hi))

        final_pos = self.forward_kinematics(q)[:3, 3]
        err_norm = float(np.linalg.norm(target - final_pos))
        return q, err_norm


class AsepticFillLineTwin:
    """ISO-5 aseptic fill-finish isolator line with cleanroom DP cascade and 6-DOF robot."""

    def __init__(self, params: Optional[FillFinishParams] = None, seed: int = 99) -> None:
        self.params = params or FillFinishParams()
        self.robot = SixDOFRobotArm(
            damping=self.params.ik_damping,
            max_iters=self.params.ik_max_iters,
            tol_m=self.params.ik_tol_m,
        )
        self._rng = np.random.default_rng(seed)

    def simulate_batch(
        self,
        vial_count: int = 24,
        door_breach_at_vial: Optional[int] = None,
    ) -> AsepticLineResult:
        """Simulate robotic vial pick-fill-place under cleanroom differential pressure control."""
        p = self.params
        dp_pa = p.nominal_dp_pa
        q_cur = np.array([0.1, 0.55, -0.45, 0.0, 0.4, 0.0], dtype=np.float64)
        vials: List[VialFillRecord] = []
        path_xyz: List[Tuple[float, float, float]] = []
        airlock_halted = False

        dt_cycle_s = 2.0
        for idx in range(vial_count):
            # Cleanroom differential pressure ODE Euler/exact step
            door_open = door_breach_at_vial is not None and idx >= door_breach_at_vial
            bleed_rate = (1.0 / p.leak_tau_s) + (p.door_leak_conductance if door_open else 0.0)
            # d(dp)/dt = hvac_supply_gain - bleed_rate * dp
            dp_eq = p.hvac_supply_gain / bleed_rate
            dp_pa = dp_eq + (dp_pa - dp_eq) * math.exp(-bleed_rate * dt_cycle_s)

            interlock_halted = dp_pa < p.iso5_min_dp_pa
            if interlock_halted:
                airlock_halted = True

            # Trajectory point on sterile conveyor / filling nest within reachable workspace
            row = idx // 6
            col = idx % 6
            target_xyz = (0.36 + 0.03 * col, -0.10 + 0.04 * row, 0.28)
            q_sol, ik_err = self.robot.solve_ik_position(target_xyz, q_init=q_cur)
            q_cur = q_sol
            fk_pos = self.robot.forward_kinematics(q_sol)[:3, 3]
            path_xyz.append((float(fk_pos[0]), float(fk_pos[1]), float(fk_pos[2])))

            if interlock_halted:
                dosed = 0.0
                accepted = False
            else:
                dosed = p.target_fill_ml + float(self._rng.normal(0.0, 0.04))
                accepted = abs(dosed - p.target_fill_ml) <= p.fill_tolerance_ml

            vials.append(
                VialFillRecord(
                    vial_index=idx,
                    dosed_volume_ml=dosed,
                    cleanroom_dp_pa=dp_pa,
                    ik_residual_m=ik_err,
                    joint_angles_rad=tuple(float(v) for v in q_sol),
                    interlock_halted=interlock_halted,
                    accepted=accepted,
                )
            )
            if interlock_halted:
                break

        dosed_vals = [v.dosed_volume_ml for v in vials if not v.interlock_halted]
        mean_fill = float(np.mean(dosed_vals)) if dosed_vals else 0.0
        min_dp = min(v.cleanroom_dp_pa for v in vials)
        max_ik = max(v.ik_residual_m for v in vials)
        acc_rate = 100.0 * sum(1 for v in vials if v.accepted) / len(vials)

        return AsepticLineResult(
            vials=vials,
            mean_fill_ml=mean_fill,
            min_dp_pa=min_dp,
            max_ik_error_m=max_ik,
            acceptance_rate_pct=acc_rate,
            airlock_breach_halted=airlock_halted,
            end_effector_path_xyz=path_xyz,
        )
