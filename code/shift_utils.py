"""Helpers for configuring temporal vs spatial shift sweeps."""

from __future__ import annotations

from typing import Iterable, List, Sequence

import numpy as np


SHIFT_MODE_CHOICES = ("time", "space")
SPACE_PROJECTION_CHOICES = ("path", "heading")


def parse_shift_values(spec: str, shift_mode: str = "time") -> List[float]:
    """Parse a comma list or start:end range into shift values."""
    if spec is None:
        return []
    text = spec.strip()
    if not text:
        return []

    cast = int if shift_mode == "time" else float
    if ":" in text:
        parts = [piece.strip() for piece in text.split(":")]
        if len(parts) not in (2, 3):
            raise ValueError(f"Invalid shift range: {spec}")
        start = cast(parts[0])
        stop = cast(parts[1])
        if len(parts) == 3:
            step = cast(parts[2])
        else:
            step = 1 if shift_mode == "time" else 1.0
        if step == 0:
            raise ValueError("Shift step cannot be zero")
        values = np.arange(start, stop, step, dtype=float if shift_mode == "space" else int)
        return values.tolist()
    return [cast(piece.strip()) for piece in text.split(",") if piece.strip()]


def build_shift_values(
    shift_mode: str,
    max_lag: int | None = None,
    lag_step: int = 1,
    max_shift_cm: float | None = None,
    shift_step_cm: float = 1.0,
    ensure_zero: bool = True,
) -> List[float]:
    """Build a symmetric shift sweep in either time steps or centimeters."""
    if shift_mode not in SHIFT_MODE_CHOICES:
      raise ValueError(f"Unsupported shift mode: {shift_mode}")

    if shift_mode == "time":
      if max_lag is None:
        raise ValueError("max_lag is required in time shift mode")
      if lag_step <= 0:
        raise ValueError("lag_step must be positive")
      values = list(range(-int(max_lag), int(max_lag) + 1, int(lag_step)))
      if ensure_zero and 0 not in values:
        values = sorted(values + [0])
      return values

    if max_shift_cm is None:
      raise ValueError("max_shift_cm is required in space shift mode")
    if shift_step_cm <= 0:
      raise ValueError("shift_step_cm must be positive")
    span = float(max_shift_cm)
    step = float(shift_step_cm)
    count = int(np.floor((2.0 * span) / step)) + 1
    values = np.linspace(-span, -span + step * (count - 1), count)
    if values.size == 0:
      values = np.array([0.0], dtype=float)
    if values[-1] < span - 1e-9:
      values = np.append(values, span)
    values = np.round(values, 8)
    if ensure_zero and not np.isclose(values, 0.0).any():
      values = np.sort(np.append(values, 0.0))
    return values.tolist()


def shift_axis_label(shift_mode: str) -> str:
    if shift_mode == "space":
      return "Position shift (cm)"
    return "Shift (time steps)"


def shift_mode_label(shift_mode: str) -> str:
    if shift_mode == "space":
      return "spatial"
    return "temporal"


def shift_config_label(shift_mode: str, space_projection: str = "path") -> str:
    if shift_mode == "space":
      if space_projection == "heading":
        return "spatial/heading"
      return "spatial/path"
    return "temporal"


def shift_values_to_cm(shift_values: Sequence[float], shift_mode: str, cm_per_step: float) -> np.ndarray:
    values = np.asarray(shift_values, dtype=float)
    if shift_mode == "space":
      return values
    return values * float(cm_per_step)


def signed_shift_threshold_mask(shift_values_cm: Iterable[float], min_shift_cm: float) -> np.ndarray:
    values = np.asarray(list(shift_values_cm), dtype=float)
    return np.abs(values) >= float(min_shift_cm)
