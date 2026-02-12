"""Helpers for consistent model and analysis output paths."""

from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MODELS_ROOT = REPO_ROOT / "Models"
ANALYSIS_ROOT = REPO_ROOT / "analysis_outputs"


def _relative_to_models(path: Path) -> Path | None:
    try:
        return path.relative_to(MODELS_ROOT)
    except ValueError:
        return None


def analysis_dir_for_checkpoint(ckpt_path: Path) -> Path:
    """Return analysis directory that mirrors the checkpoint location under Models/."""
    ckpt_path = Path(ckpt_path).resolve()
    ckpt_dir = ckpt_path.parent if ckpt_path.is_file() else ckpt_path
    rel = _relative_to_models(ckpt_dir)
    if rel is None:
        return ANALYSIS_ROOT / ckpt_dir.name
    return ANALYSIS_ROOT / rel


def model_name_from_checkpoint(ckpt_path: Path) -> str:
    ckpt_path = Path(ckpt_path).resolve()
    rel = _relative_to_models(ckpt_path if ckpt_path.is_dir() else ckpt_path.parent)
    if rel is None or not rel.parts:
        return "unknown_model"
    return rel.parts[0]


def analysis_summary_dir(model_name: str, subdir: str | None = None) -> Path:
    base = ANALYSIS_ROOT / model_name / "summary"
    return base / subdir if subdir else base
