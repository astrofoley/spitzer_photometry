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
    After applying the projected support trim (mask_cov), w_data must be
    recomputed/masked before any H/rhs terms use it.
    """
    src = Path(solver.__file__).read_text(encoding="utf-8")
    anchor = "mask = mask & mask_cov"
    assert anchor in src, "Expected support-trim mask step missing in solver."
    after = src.split(anchor, 1)[1]
    first_use = after.find("w_data")
    assert first_use >= 0, "Expected w_data use after mask trim not found."
    window = after[: first_use + 120]
    has_weight_update = (
        ("w_data[~mask" in window)
        or ("w_data *= mask" in window)
        or ("w_data[mask.flatten()]" in window)
    )
    assert has_weight_update, (
        "w_data is used after mask_cov trim without explicit weight remasking/rebuild; "
        "trimmed pixels may still contribute to H/rhs."
    )


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
    Validate that current solver has exact-vs-approx LTWL branches and current
    config default prefers approximate branch for non-trivial scenes.
    """
    curr = Path(solver.__file__).read_text(encoding="utf-8")
    assert "PRF_GLS_LTWL_DIAG_MAX_PIXELS" in curr
    assert "if ltwl_cap > 0 and n_scene <= ltwl_cap" in curr
    assert "w_scene_diag = _project_native_to_scene" in curr
    assert int(getattr(config, "PRF_GLS_LTWL_DIAG_MAX_PIXELS", -1)) == 0
