"""IEC 61131-3 standard stateful Function Blocks (Timers, Counters, Triggers, PID)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


class FunctionBlock:
    """Base interface for stateful IEC 61131-3 function blocks."""

    def call(self, params: Dict[str, Any], dt_s: float) -> None:
        raise NotImplementedError

    def read_field(self, field_name: str) -> Any:
        if not hasattr(self, field_name):
            raise KeyError(f"Function block {type(self).__name__} has no field {field_name!r}")
        return getattr(self, field_name)


@dataclass
class TON(FunctionBlock):
    """On-delay timer: Q becomes TRUE after IN stays TRUE for at least PT seconds."""

    IN: bool = False
    PT: float = 0.0
    Q: bool = False
    ET: float = 0.0

    def call(self, params: Dict[str, Any], dt_s: float) -> None:
        self.IN = bool(params.get("IN", self.IN))
        if "PT" in params:
            self.PT = max(0.0, float(params["PT"]))
        if not self.IN:
            self.ET = 0.0
            self.Q = False
        else:
            self.ET = min(self.PT, self.ET + dt_s) if self.PT > 0.0 else 0.0
            self.Q = self.ET >= self.PT


@dataclass
class TOF(FunctionBlock):
    """Off-delay timer: Q drops to FALSE after IN stays FALSE for PT seconds."""

    IN: bool = False
    PT: float = 0.0
    Q: bool = False
    ET: float = 0.0
    _prev_in: bool = False

    def call(self, params: Dict[str, Any], dt_s: float) -> None:
        self.IN = bool(params.get("IN", self.IN))
        if "PT" in params:
            self.PT = max(0.0, float(params["PT"]))
        if self.IN:
            self.Q = True
            self.ET = 0.0
        else:
            if self._prev_in or self.Q:
                self.ET = min(self.PT, self.ET + dt_s)
                self.Q = self.ET < self.PT
        self._prev_in = self.IN


@dataclass
class TP(FunctionBlock):
    """Pulse timer: rising edge on IN generates fixed-duration PT pulse on Q."""

    IN: bool = False
    PT: float = 0.0
    Q: bool = False
    ET: float = 0.0
    _prev_in: bool = False

    def call(self, params: Dict[str, Any], dt_s: float) -> None:
        self.IN = bool(params.get("IN", self.IN))
        if "PT" in params:
            self.PT = max(0.0, float(params["PT"]))
        rising = self.IN and not self._prev_in
        if not self.Q and rising and self.PT > 0.0:
            self.ET = min(self.PT, dt_s)
            self.Q = self.ET < self.PT
        elif self.Q:
            self.ET += dt_s
            if self.ET >= self.PT:
                self.Q = False
                self.ET = self.PT
        elif not self.IN:
            self.ET = 0.0
        self._prev_in = self.IN


@dataclass
class CTU(FunctionBlock):
    """Up-counter with rising-edge trigger CU and asynchronous reset R."""

    CU: bool = False
    R: bool = False
    PV: int = 0
    Q: bool = False
    CV: int = 0
    _prev_cu: bool = False

    def call(self, params: Dict[str, Any], dt_s: float) -> None:
        del dt_s
        self.CU = bool(params.get("CU", self.CU))
        self.R = bool(params.get("R", self.R))
        if "PV" in params:
            self.PV = int(params["PV"])
        if self.R:
            self.CV = 0
        elif self.CU and not self._prev_cu:
            self.CV += 1
        self.Q = self.CV >= self.PV
        self._prev_cu = self.CU


@dataclass
class CTD(FunctionBlock):
    """Down-counter with rising-edge trigger CD and preset load LD."""

    CD: bool = False
    LD: bool = False
    PV: int = 0
    Q: bool = False
    CV: int = 0
    _prev_cd: bool = False

    def call(self, params: Dict[str, Any], dt_s: float) -> None:
        del dt_s
        self.CD = bool(params.get("CD", self.CD))
        self.LD = bool(params.get("LD", self.LD))
        if "PV" in params:
            self.PV = int(params["PV"])
        if self.LD:
            self.CV = self.PV
        elif self.CD and not self._prev_cd and self.CV > 0:
            self.CV -= 1
        self.Q = self.CV <= 0
        self._prev_cd = self.CD


@dataclass
class R_TRIG(FunctionBlock):
    """Rising-edge detector."""

    CLK: bool = False
    Q: bool = False
    _prev: bool = False

    def call(self, params: Dict[str, Any], dt_s: float) -> None:
        del dt_s
        self.CLK = bool(params.get("CLK", self.CLK))
        self.Q = self.CLK and not self._prev
        self._prev = self.CLK


@dataclass
class F_TRIG(FunctionBlock):
    """Falling-edge detector."""

    CLK: bool = False
    Q: bool = False
    _prev: bool = True

    def call(self, params: Dict[str, Any], dt_s: float) -> None:
        del dt_s
        self.CLK = bool(params.get("CLK", self.CLK))
        self.Q = (not self.CLK) and self._prev
        self._prev = self.CLK


@dataclass
class PID_FB(FunctionBlock):
    """Industrial anti-windup PID controller with derivative-on-measurement and bumpless transfer."""

    PV: float = 0.0
    SP: float = 0.0
    KP: float = 1.0
    KI: float = 0.0
    KD: float = 0.0
    OUT_MIN: float = 0.0
    OUT_MAX: float = 100.0
    MANUAL: bool = False
    MANUAL_VAL: float = 0.0
    OUT: float = 0.0
    ERR: float = 0.0
    _integral: float = 0.0
    _prev_pv: float = 0.0
    _initialized: bool = False

    def call(self, params: Dict[str, Any], dt_s: float) -> None:
        self.PV = float(params.get("PV", self.PV))
        self.SP = float(params.get("SP", self.SP))
        self.KP = float(params.get("KP", self.KP))
        self.KI = float(params.get("KI", self.KI))
        self.KD = float(params.get("KD", self.KD))
        self.OUT_MIN = float(params.get("OUT_MIN", self.OUT_MIN))
        self.OUT_MAX = float(params.get("OUT_MAX", self.OUT_MAX))
        self.MANUAL = bool(params.get("MANUAL", self.MANUAL))
        self.MANUAL_VAL = float(params.get("MANUAL_VAL", self.MANUAL_VAL))

        if not self._initialized:
            self._prev_pv = self.PV
            self._initialized = True

        self.ERR = self.SP - self.PV
        if self.MANUAL:
            clamped = max(self.OUT_MIN, min(self.OUT_MAX, self.MANUAL_VAL))
            # Bumpless tracking of integral term while in manual mode
            self._integral = clamped - self.KP * self.ERR
            self.OUT = clamped
            self._prev_pv = self.PV
            return

        p_term = self.KP * self.ERR
        d_term = 0.0
        if dt_s > 0.0:
            d_term = -self.KD * (self.PV - self._prev_pv) / dt_s

        candidate_i = self._integral + self.KI * self.ERR * dt_s
        unclamped = p_term + candidate_i + d_term
        clamped = max(self.OUT_MIN, min(self.OUT_MAX, unclamped))

        # Conditional integration (anti-windup): freeze integrator if saturated in error direction
        if unclamped == clamped or (unclamped > self.OUT_MAX and self.ERR < 0.0) or (
            unclamped < self.OUT_MIN and self.ERR > 0.0
        ):
            self._integral = candidate_i

        self.OUT = clamped
        self._prev_pv = self.PV


FUNCTION_BLOCK_REGISTRY = {
    "TON": TON,
    "TOF": TOF,
    "TP": TP,
    "CTU": CTU,
    "CTD": CTD,
    "R_TRIG": R_TRIG,
    "F_TRIG": F_TRIG,
    "PID_FB": PID_FB,
}
