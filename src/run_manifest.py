"""Write `run_manifest.json` for reproducibility (git, env, config snapshot, inputs)."""
from __future__ import annotations

import importlib.metadata
import json
import os
import subprocess
import sys
from typing import Any, Mapping, Sequence

import numpy as np

from . import config


def _git_commit_id() -> str | None:
    try:
        p = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=config.BASE_DIR,
            capture_output=True,
            text=True,
            timeout=8,
            check=True,
        )
        return p.stdout.strip() or None
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None


def _git_dirty() -> bool | None:
    try:
        p = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=config.BASE_DIR,
            capture_output=True,
            text=True,
            timeout=8,
            check=False,
        )
        if p.returncode != 0:
            return None
        return bool(p.stdout.strip())
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None


def _json_safe(x: Any) -> Any:
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (list, tuple)):
        return [_json_safe(v) for v in x]
    if isinstance(x, dict):
        return {str(k): _json_safe(v) for k, v in x.items()}
    return str(x)


def config_snapshot() -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name in dir(config):
        if name.startswith("_"):
            continue
        val = getattr(config, name)
        if callable(val):
            continue
        js = _json_safe(val)
        try:
            json.dumps(js)
            out[name] = js
        except (TypeError, ValueError):
            out[name] = str(val)
    return out


def package_versions(keys: Sequence[str] | None = None) -> dict[str, str | None]:
    names = list(keys) if keys is not None else [
        "numpy",
        "scipy",
        "astropy",
        "matplotlib",
        "sep",
        "reproject",
    ]
    out: dict[str, str | None] = {}
    for n in names:
        try:
            out[n] = importlib.metadata.version(n)
        except importlib.metadata.PackageNotFoundError:
            out[n] = None
    return out


def build_run_manifest(
    input_image_paths: Sequence[str],
    *,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    paths = sorted({os.path.abspath(p) for p in input_image_paths})
    doc: dict[str, Any] = {
        "schema": "spitzer_photometry.run_manifest.v1",
        "git_commit": _git_commit_id(),
        "git_dirty": _git_dirty(),
        "python": sys.version.split("\n")[0],
        "executable": sys.executable,
        "packages": package_versions(),
        "config": config_snapshot(),
        "input_cbcd_paths": paths,
        "n_input_frames": len(paths),
    }
    if extra:
        doc["extra"] = _json_safe(dict(extra))
    return doc


def write_run_manifest(
    path: str | os.PathLike[str],
    input_image_paths: Sequence[str],
    *,
    extra: Mapping[str, Any] | None = None,
) -> str:
    """Write manifest JSON; returns the absolute path written."""
    dest = os.path.abspath(path)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    doc = build_run_manifest(input_image_paths, extra=extra)
    with open(dest, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2, sort_keys=False)
        f.write("\n")
    return dest

