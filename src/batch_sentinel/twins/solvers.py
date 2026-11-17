"""Adaptive Cash-Karp Runge-Kutta-Fehlberg (RKF45) ODE solver for digital twins."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray


ODEFunc = Callable[[float, NDArray[np.float64]], NDArray[np.float64]]


@dataclass(frozen=True)
class ODEResult:
    """Output trajectory and solver diagnostics from :func:`integrate_rkf45`."""

    t: NDArray[np.float64]
    y: NDArray[np.float64]
    accepted_steps: int
    rejected_steps: int
    max_error_ratio: float


# Cash-Karp Butcher tableau coefficients
_A2, _A3, _A4, _A5, _A6 = 1.0 / 5.0, 3.0 / 10.0, 3.0 / 5.0, 1.0, 7.0 / 8.0
_B21 = 1.0 / 5.0
_B31, _B32 = 3.0 / 40.0, 9.0 / 40.0
_B41, _B42, _B43 = 3.0 / 10.0, -9.0 / 10.0, 6.0 / 5.0
_B51, _B52, _B53, _B54 = -11.0 / 54.0, 5.0 / 2.0, -70.0 / 27.0, 35.0 / 27.0
_B61, _B62, _B63, _B64, _B65 = (
    1631.0 / 55296.0,
    175.0 / 512.0,
    575.0 / 13824.0,
    44275.0 / 110592.0,
    253.0 / 4096.0,
)
_C1, _C3, _C4, _C6 = 37.0 / 378.0, 250.0 / 621.0, 125.0 / 594.0, 512.0 / 1771.0
_DC1 = _C1 - 2825.0 / 27648.0
_DC3 = _C3 - 18575.0 / 48384.0
_DC4 = _C4 - 13525.0 / 55296.0
_DC5 = -277.0 / 14336.0
_DC6 = _C6 - 1.0 / 4.0


def integrate_rkf45(
    rhs: ODEFunc,
    t_span: Tuple[float, float],
    y0: NDArray[np.float64],
    rtol: float = 1e-6,
    atol: float = 1e-8,
    initial_dt: Optional[float] = None,
    min_dt: float = 1e-9,
    max_dt: Optional[float] = None,
    non_negative_indices: Optional[Tuple[int, ...]] = None,
    max_steps: int = 50000,
) -> ODEResult:
    """Integrate ``dy/dt = rhs(t, y)`` over ``t_span`` using adaptive RKF45.

    Args:
        rhs: Derivative callback returning a 1D float64 array matching ``y0``.
        t_span: ``(t_start, t_end)`` with ``t_end >= t_start``.
        y0: Initial state vector.
        rtol: Relative local truncation error tolerance.
        atol: Absolute local truncation error tolerance.
        initial_dt: Optional initial step guess.
        min_dt: Minimum allowed step size before accepting with warning floor.
        max_dt: Maximum allowed step size.
        non_negative_indices: Optional state indices clamped to ``>= 0.0``.
        max_steps: Hard upper limit on step evaluations to prevent infinite loops.
    """
    t0, t1 = float(t_span[0]), float(t_span[1])
    if t1 < t0:
        raise ValueError(f"t_span end {t1} must be >= start {t0}")

    y = np.array(y0, dtype=np.float64, copy=True)
    if t1 == t0:
        return ODEResult(
            t=np.array([t0], dtype=np.float64),
            y=y[None, :],
            accepted_steps=0,
            rejected_steps=0,
            max_error_ratio=0.0,
        )

    dt = initial_dt if initial_dt is not None else min(0.05 * (t1 - t0), 0.25)
    dt = max(dt, min_dt)
    if max_dt is not None:
        dt = min(dt, max_dt)

    t_list: List[float] = [t0]
    y_list: List[NDArray[np.float64]] = [y.copy()]

    t = t0
    accepted = 0
    rejected = 0
    max_err_ratio = 0.0

    for _ in range(max_steps):
        if t >= t1 - 1e-15:
            break
        if t + dt > t1:
            dt = t1 - t

        k1 = rhs(t, y)
        k2 = rhs(t + _A2 * dt, y + dt * _B21 * k1)
        k3 = rhs(t + _A3 * dt, y + dt * (_B31 * k1 + _B32 * k2))
        k4 = rhs(t + _A4 * dt, y + dt * (_B41 * k1 + _B42 * k2 + _B43 * k3))
        k5 = rhs(t + _A5 * dt, y + dt * (_B51 * k1 + _B52 * k2 + _B53 * k3 + _B54 * k4))
        k6 = rhs(
            t + _A6 * dt,
            y + dt * (_B61 * k1 + _B62 * k2 + _B63 * k3 + _B64 * k4 + _B65 * k5),
        )

        y_next = y + dt * (_C1 * k1 + _C3 * k3 + _C4 * k4 + _C6 * k6)
        err_vec = dt * (_DC1 * k1 + _DC3 * k3 + _DC4 * k4 + _DC5 * k5 + _DC6 * k6)
        scale = atol + rtol * np.maximum(np.abs(y), np.abs(y_next))
        err_ratio = float(np.max(np.abs(err_vec) / scale))

        if err_ratio <= 1.0 or dt <= min_dt:
            t += dt
            y = y_next
            if non_negative_indices:
                for idx in non_negative_indices:
                    if y[idx] < 0.0:
                        y[idx] = 0.0
            accepted += 1
            max_err_ratio = max(max_err_ratio, err_ratio)
            t_list.append(t)
            y_list.append(y.copy())

            if err_ratio > 1e-16:
                factor = min(4.0, 0.9 * (err_ratio ** -0.2))
            else:
                factor = 4.0
            dt = max(min_dt, dt * factor)
            if max_dt is not None:
                dt = min(dt, max_dt)
        else:
            rejected += 1
            factor = max(0.1, 0.9 * (err_ratio ** -0.25))
            dt = max(min_dt, dt * factor)

    return ODEResult(
        t=np.asarray(t_list, dtype=np.float64),
        y=np.vstack(y_list),
        accepted_steps=accepted,
        rejected_steps=rejected,
        max_error_ratio=max_err_ratio,
    )
