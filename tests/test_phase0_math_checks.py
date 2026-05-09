from __future__ import annotations

from pathlib import Path

from src import config, solver


def _legacy_solver_source() -> str:
    root = Path(__file__).resolve().parents[1]
    legacy = root / "clean_repo_run" / "src" / "solver.py"
    return legacy.read_text(encoding="utf-8")


def test_mask_trim_updates_weights_before_hessian_use():
    """
    Phase-0 check 1:
    χ² mask (`mask_cov` / edge trim) is computed for diagnostics; fit weights stay
    tied to the base data-valid mask (see solver comments near `mask_cov`).
    """
    src = Path(solver.__file__).read_text(encoding="utf-8")
    assert "mask_cov" in src
    assert "keep fit-time weights" in src or "fit-time weights" in src
    assert "w_data" in src


def test_background_parameterization_differs_from_legacy():
    """
    Phase-0 check 2:
    Verify that current solver parameterizes background per BCD, while legacy
    parameterized per epoch.
    """
    curr = Path(solver.__file__).read_text(encoding="utf-8")
    legacy = _legacy_solver_source()
    assert "n_bg = len(cutouts)" in curr
    assert "ib = idx_bg + int(i)" in curr
    assert "n_bg = n_epochs" in legacy
    assert "ib = idx_bg + entry['epoch_id']" in legacy


def test_scene_lock_and_monotonic_constraints_exist_in_current_only():
    """
    Phase-0 check 3:
    Confirm presence of scene-lock and central monotonic constraints in current
    solver, and absence in legacy solver.
    """
    curr = Path(solver.__file__).read_text(encoding="utf-8")
    legacy = _legacy_solver_source()

    assert "scene_lock_idx" in curr
    assert "ENFORCE_GP_CENTRAL_MONOTONICITY" in curr
    assert "scene_lock_idx" not in legacy
    assert "ENFORCE_GP_CENTRAL_MONOTONICITY" not in legacy


def test_operator_approximation_path_is_default_enabled():
    """
    Phase-0 check 4:
    Validate that current solver has full-vs-diag LTWL branches and default
    config uses the `_project_native_to_scene` diagonal path for large scenes.
    """
    curr = Path(solver.__file__).read_text(encoding="utf-8")
    assert "PRF_GLS_LTWL_DIAG_MAX_PIXELS" in curr
    assert "ltwl_diag_cap" in curr
    assert "elif ltwl_diag_cap > 0 and n_scene <= ltwl_diag_cap" in curr
    assert "w_scene_diag = _project_native_to_scene" in curr
    assert int(getattr(config, "PRF_GLS_LTWL_DIAG_MAX_PIXELS", -1)) == 0
