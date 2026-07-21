"""Reproducibility FREEZE / provenance manifest generator.

Captures everything needed to reproduce the PGC-rigor pipeline into a single
deterministic manifest (``reproducibility/manifest.json``) plus a human-readable
``reproducibility/FREEZE.md``:

  * git commit hash, branch, dirty flag + list of changed files
  * python version, platform, and the FULL ``pip freeze`` of the project venv
  * per-checkpoint sha256 + byte size, and the training config parsed out of the
    ``run_ID`` folder name (steps_/batch_/RNN_/Ng/... a la utils.generate_run_ID)
  * the exact analysis config used by the rigor pipeline: the dataclass fields of
    ``pgc_classifier.ClassifierConfig`` and the ``pgc_covariates`` defaults
  * a caller-supplied timestamp -- this module NEVER calls datetime.now() /
    time.time(); the frozen timestamp is passed in (or read from FREEZE_TIMESTAMP)
    so the manifest is deterministic given identical inputs.

Importable (``from pgc_freeze import freeze_manifest``) and CLI-runnable.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import platform
import re
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

_CODE_DIR = os.path.dirname(os.path.abspath(__file__))
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

from path_utils import REPO_ROOT, MODELS_ROOT  # noqa: E402

DEFAULT_VENV_PY = str(REPO_ROOT / ".venv" / "bin" / "python")
REPRO_DIR = REPO_ROOT / "reproducibility"
MANIFEST_PATH = REPRO_DIR / "manifest.json"
FREEZE_MD_PATH = REPRO_DIR / "FREEZE.md"
SCHEMA = "pgc_freeze/1"


# --------------------------------------------------------------------------- #
# subprocess helpers
# --------------------------------------------------------------------------- #
def _run(cmd, cwd=None) -> tuple[int, str, str]:
    """Run ``cmd`` (list) and return (returncode, stdout, stderr) as text."""
    try:
        p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
        return p.returncode, p.stdout, p.stderr
    except Exception as exc:  # pragma: no cover - defensive
        return 1, "", f"{type(exc).__name__}: {exc}"


# --------------------------------------------------------------------------- #
# git provenance
# --------------------------------------------------------------------------- #
def git_info(repo_root: Path = REPO_ROOT) -> dict:
    repo_root = str(repo_root)
    rc_hash, out_hash, _ = _run(["git", "rev-parse", "HEAD"], cwd=repo_root)
    rc_br, out_br, _ = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=repo_root)
    rc_st, out_st, _ = _run(["git", "status", "--porcelain"], cwd=repo_root)

    changed = []
    if rc_st == 0:
        for line in out_st.splitlines():
            if not line.strip():
                continue
            # porcelain: 2-char status, space, path (path may be quoted)
            status = line[:2]
            path = line[3:]
            changed.append({"status": status.strip(), "path": path})

    return {
        "available": rc_hash == 0,
        "commit": out_hash.strip() if rc_hash == 0 else None,
        "branch": out_br.strip() if rc_br == 0 else None,
        "dirty": bool(changed),
        "n_changed": len(changed),
        "changed_files": changed,
    }


# --------------------------------------------------------------------------- #
# environment provenance
# --------------------------------------------------------------------------- #
def pip_freeze(venv_python: str = DEFAULT_VENV_PY) -> list[str]:
    rc, out, _ = _run([venv_python, "-m", "pip", "freeze"])
    if rc != 0:
        return []
    return [ln.strip() for ln in out.splitlines() if ln.strip()]


def env_info(venv_python: str = DEFAULT_VENV_PY) -> dict:
    rc, out, _ = _run([venv_python, "--version"])
    venv_ver = out.strip() if rc == 0 else None
    if not venv_ver:
        # some pythons print --version to stderr
        rc2, _, err = _run([venv_python, "--version"])
        venv_ver = err.strip() if err.strip() else None
    freeze = pip_freeze(venv_python)
    return {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "sys_version": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "venv_python": venv_python,
        "venv_python_version": venv_ver,
        "pip_freeze": freeze,
        "n_packages": len(freeze),
    }


# --------------------------------------------------------------------------- #
# checkpoint provenance
# --------------------------------------------------------------------------- #
def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


# labeled single-token key -> config field name (values are strings, decimals
# stripped by generate_run_ID's .replace('.', ''))
_RUN_ID_LABELED = {
    "steps": "sequence_length",
    "batch": "batch_size",
    "rf": "place_cell_rf",
    "DoG": "DoG",
    "periodic": "periodic",
    "lr": "learning_rate",
    "rank": "rank",
}


def parse_run_ID(name: str) -> dict:
    """Best-effort inverse of utils.generate_run_ID.

    Template produced by generate_run_ID:
      steps_{seq}_batch_{bs}_{RNN_type}_{Ng}_{activation}_rf_{rf}_DoG_{DoG}
      _periodic_{periodic}_lr_{lr}_weight_decay_{wd}[_rank_{rank}]

    Returns whatever fields can be recovered; ``{}`` if the name is clearly not a
    run_ID. Note generate_run_ID strips '.' so decimals appear as e.g. rf_012.
    """
    if not name or "steps_" not in name:
        return {}
    cfg: dict = {}
    for key, field in _RUN_ID_LABELED.items():
        m = re.search(rf"(?:^|_){re.escape(key)}_([^_]+)", name)
        if m:
            cfg[field] = m.group(1)
    # weight_decay is a two-token label
    m = re.search(r"weight_decay_([^_]+)", name)
    if m:
        cfg["weight_decay"] = m.group(1)
    # positional RNN_type / Ng / activation live between "batch_<val>_" and "_rf_"
    m = re.search(r"batch_[^_]+_(.+?)_rf_", name)
    if m:
        mid = m.group(1).split("_")
        if len(mid) >= 3:
            cfg["activation"] = mid[-1]
            cfg["Ng"] = mid[-2]
            cfg["RNN_type"] = "_".join(mid[:-2])
        elif mid:
            cfg["RNN_type"] = mid[0]
    return cfg


def find_run_ID_dir(ckpt_path: Path) -> str | None:
    """Return the nearest ancestor folder name that looks like a run_ID."""
    ckpt_path = Path(ckpt_path).resolve()
    start = ckpt_path.parent if ckpt_path.is_file() else ckpt_path
    for parent in [start, *start.parents]:
        if parent.name.startswith("steps_"):
            return parent.name
    return None


def checkpoint_record(ckpt: str | Path) -> dict:
    path = Path(ckpt)
    resolved = path.resolve()
    rec: dict = {
        "path": str(path),
        "resolved_path": str(resolved),
        "exists": resolved.is_file(),
    }
    try:
        rel = resolved.relative_to(MODELS_ROOT.resolve())
        rec["relative_to_models"] = str(rel)
    except ValueError:
        rec["relative_to_models"] = None

    if not resolved.is_file():
        rec["error"] = "checkpoint file not found"
        rec["sha256"] = None
        rec["size_bytes"] = None
        rec["run_ID"] = None
        rec["training_config"] = None
        return rec

    rec["size_bytes"] = resolved.stat().st_size
    rec["sha256"] = sha256_file(resolved)
    run_id = find_run_ID_dir(resolved)
    rec["run_ID"] = run_id
    rec["training_config"] = parse_run_ID(run_id) if run_id else None
    return rec


# --------------------------------------------------------------------------- #
# analysis config provenance
# --------------------------------------------------------------------------- #
def analysis_config() -> dict:
    """Capture the exact defaults driving the rigor pipeline."""
    out: dict = {}
    try:
        import pgc_classifier as PC  # noqa: E402
        out["pgc_classifier.ClassifierConfig"] = asdict(PC.ClassifierConfig())
    except Exception as exc:  # pragma: no cover - defensive
        out["pgc_classifier.ClassifierConfig"] = {"error": f"{type(exc).__name__}: {exc}"}
    try:
        import pgc_covariates as CV  # noqa: E402
        sig = inspect.signature(CV.assemble_covariates)
        defaults = {
            name: (p.default if p.default is not inspect._empty else None)
            for name, p in sig.parameters.items()
            if name != "lm"
        }
        out["pgc_covariates.assemble_covariates_defaults"] = defaults
    except Exception as exc:  # pragma: no cover - defensive
        out["pgc_covariates.assemble_covariates_defaults"] = {
            "error": f"{type(exc).__name__}: {exc}"
        }
    return out


# --------------------------------------------------------------------------- #
# timestamp (deterministic; never wall-clock)
# --------------------------------------------------------------------------- #
def resolve_timestamp(ts: str | None) -> dict:
    """Resolve the frozen timestamp WITHOUT touching the wall clock.

    Order: explicit value -> FREEZE_TIMESTAMP env var -> unspecified sentinel.
    """
    if ts:
        return {"value": ts, "source": "supplied"}
    env = os.environ.get("FREEZE_TIMESTAMP")
    if env:
        return {"value": env, "source": "FREEZE_TIMESTAMP env"}
    return {"value": None, "source": "UNSPECIFIED"}


# --------------------------------------------------------------------------- #
# manifest assembly
# --------------------------------------------------------------------------- #
def freeze_manifest(outputs: dict, extra: dict | None = None) -> dict:
    """Build the reproducibility manifest dict.

    Parameters
    ----------
    outputs : dict
        Recognized keys:
          ``timestamp``    -> caller-supplied frozen timestamp string
          ``checkpoints``  -> iterable of checkpoint paths to fingerprint
          ``figures``      -> {figure_name: {checkpoint, script, params, ...}}
          ``venv_python``  -> python interpreter whose pip freeze to capture
    extra : dict, optional
        Arbitrary provenance merged under ``manifest['extra']``.
    """
    outputs = dict(outputs or {})
    extra = dict(extra or {})

    ts = resolve_timestamp(outputs.get("timestamp"))
    venv_python = outputs.get("venv_python") or DEFAULT_VENV_PY
    ckpts = list(outputs.get("checkpoints") or [])
    figures = dict(outputs.get("figures") or {})

    manifest = {
        "schema": SCHEMA,
        "timestamp": ts["value"],
        "timestamp_source": ts["source"],
        "repo_root": str(REPO_ROOT),
        "git": git_info(),
        "environment": env_info(venv_python),
        "checkpoints": [checkpoint_record(c) for c in ckpts],
        "analysis_config": analysis_config(),
        "figures": figures,
        "extra": extra,
    }
    return manifest


# --------------------------------------------------------------------------- #
# checkpoint discovery
# --------------------------------------------------------------------------- #
def discover_checkpoints(models_root: str | Path, glob: str = "**/*.pth") -> list[str]:
    root = Path(models_root)
    if not root.exists():
        return []
    return [str(p) for p in sorted(root.glob(glob)) if p.is_file()]


# --------------------------------------------------------------------------- #
# FREEZE.md rendering
# --------------------------------------------------------------------------- #
def _md_kv_table(d: dict) -> str:
    lines = ["| key | value |", "| --- | --- |"]
    for k, v in d.items():
        lines.append(f"| `{k}` | `{v}` |")
    return "\n".join(lines)


def render_freeze_md(manifest: dict) -> str:
    g = manifest.get("git", {})
    env = manifest.get("environment", {})
    out: list[str] = []
    out.append("# Reproducibility FREEZE")
    out.append("")
    out.append(f"- **Frozen timestamp:** `{manifest.get('timestamp')}` "
               f"(source: {manifest.get('timestamp_source')})")
    out.append(f"- **Schema:** `{manifest.get('schema')}`")
    out.append(f"- **Repo root:** `{manifest.get('repo_root')}`")
    out.append("")

    out.append("## Git")
    out.append("")
    out.append(f"- **Commit:** `{g.get('commit')}`")
    out.append(f"- **Branch:** `{g.get('branch')}`")
    out.append(f"- **Dirty:** `{g.get('dirty')}` ({g.get('n_changed')} changed files)")
    if g.get("changed_files"):
        out.append("")
        out.append("<details><summary>Changed files</summary>")
        out.append("")
        for cf in g["changed_files"]:
            out.append(f"- `{cf['status']}` {cf['path']}")
        out.append("")
        out.append("</details>")
    out.append("")

    out.append("## Environment")
    out.append("")
    out.append(f"- **Python:** `{env.get('python_version')}` "
               f"({env.get('python_implementation')})")
    out.append(f"- **Platform:** `{env.get('platform')}`")
    out.append(f"- **venv python:** `{env.get('venv_python')}` "
               f"(`{env.get('venv_python_version')}`)")
    out.append(f"- **Packages (pip freeze):** {env.get('n_packages')}")
    out.append("")
    out.append("<details><summary>pip freeze</summary>")
    out.append("")
    out.append("```")
    out.extend(env.get("pip_freeze", []))
    out.append("```")
    out.append("")
    out.append("</details>")
    out.append("")

    out.append("## Checkpoints")
    out.append("")
    for rec in manifest.get("checkpoints", []):
        out.append(f"### `{rec.get('path')}`")
        out.append("")
        out.append(f"- **sha256:** `{rec.get('sha256')}`")
        out.append(f"- **size (bytes):** `{rec.get('size_bytes')}`")
        out.append(f"- **run_ID:** `{rec.get('run_ID')}`")
        tc = rec.get("training_config")
        if tc:
            out.append("- **training config (parsed from run_ID):**")
            out.append("")
            out.append(_md_kv_table(tc))
        if rec.get("error"):
            out.append(f"- **error:** {rec['error']}")
        out.append("")

    out.append("## Analysis config")
    out.append("")
    for name, cfg in manifest.get("analysis_config", {}).items():
        out.append(f"### `{name}`")
        out.append("")
        out.append(_md_kv_table(cfg))
        out.append("")

    figures = manifest.get("figures", {})
    if figures:
        out.append("## Figures / results")
        out.append("")
        for fig_name, spec in figures.items():
            out.append(f"### {fig_name}")
            out.append("")
            if isinstance(spec, dict):
                out.append(_md_kv_table(spec))
            else:
                out.append(f"`{spec}`")
            out.append("")

    extra = manifest.get("extra", {})
    if extra:
        out.append("## Extra")
        out.append("")
        out.append(_md_kv_table(extra))
        out.append("")

    return "\n".join(out) + "\n"


# --------------------------------------------------------------------------- #
# writing
# --------------------------------------------------------------------------- #
def write_manifest(manifest: dict,
                   manifest_path: Path = MANIFEST_PATH,
                   md_path: Path = FREEZE_MD_PATH) -> tuple[Path, Path]:
    manifest_path = Path(manifest_path)
    md_path = Path(md_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=False)
        fh.write("\n")
    with open(md_path, "w") as fh:
        fh.write(render_freeze_md(manifest))
    return manifest_path, md_path


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser(
        description="Generate a deterministic reproducibility manifest + FREEZE.md")
    p.add_argument("--checkpoints", nargs="*", default=[],
                   help="explicit checkpoint .pth paths to fingerprint")
    p.add_argument("--models_root", default=None,
                   help="discover checkpoints under this dir (in addition to --checkpoints)")
    p.add_argument("--glob", default="**/*.pth",
                   help="glob used with --models_root discovery")
    p.add_argument("--timestamp", default=os.environ.get("FREEZE_TIMESTAMP"),
                   help="frozen timestamp string (default: env FREEZE_TIMESTAMP). "
                        "This module never reads the wall clock.")
    p.add_argument("--figures", default=None,
                   help="path to a JSON mapping figure_name -> {checkpoint, script, params}")
    p.add_argument("--venv_python", default=DEFAULT_VENV_PY,
                   help="python interpreter whose pip freeze to capture")
    p.add_argument("--out_dir", default=str(REPRO_DIR),
                   help="output directory for manifest.json + FREEZE.md")
    p.add_argument("--extra", default=None,
                   help="path to a JSON file merged under manifest['extra']")
    args = p.parse_args()

    checkpoints = list(args.checkpoints)
    if args.models_root:
        checkpoints.extend(discover_checkpoints(args.models_root, args.glob))
    # de-dup preserving order
    seen = set()
    checkpoints = [c for c in checkpoints if not (c in seen or seen.add(c))]

    figures = {}
    if args.figures:
        with open(args.figures) as fh:
            figures = json.load(fh)

    extra = {}
    if args.extra:
        with open(args.extra) as fh:
            extra = json.load(fh)

    outputs = {
        "timestamp": args.timestamp,
        "checkpoints": checkpoints,
        "figures": figures,
        "venv_python": args.venv_python,
    }
    manifest = freeze_manifest(outputs, extra)

    out_dir = Path(args.out_dir)
    manifest_path, md_path = write_manifest(
        manifest, out_dir / "manifest.json", out_dir / "FREEZE.md")

    g = manifest["git"]
    env = manifest["environment"]
    print("=== reproducibility freeze ===")
    print(f"manifest keys      : {list(manifest.keys())}")
    print(f"timestamp          : {manifest['timestamp']} ({manifest['timestamp_source']})")
    print(f"git.commit         : {g['commit']}")
    print(f"git.branch         : {g['branch']}  dirty={g['dirty']} "
          f"(n_changed={g['n_changed']})")
    print(f"env.python_version : {env['python_version']}")
    print(f"env.n_packages     : {env['n_packages']} (pip freeze captured)")
    print(f"checkpoints        : {len(manifest['checkpoints'])}")
    for rec in manifest["checkpoints"]:
        print(f"  - {rec['path']}")
        print(f"      sha256 = {rec['sha256']}")
        print(f"      size   = {rec['size_bytes']} bytes  run_ID={rec['run_ID']}")
        if rec.get("training_config"):
            print(f"      training_config = {rec['training_config']}")
    print(f"analysis_config    : {list(manifest['analysis_config'].keys())}")
    print(f"figures            : {list(manifest['figures'].keys())}")
    print(f"wrote manifest -> {manifest_path}")
    print(f"wrote FREEZE   -> {md_path}")


if __name__ == "__main__":
    main()
