"""Diagnostic context: treat PRF convolution as identity on scene and native grids.

Use this to verify scale consistency by comparing fits with full PRF vs projection-only
(forward **F ≈ P** when ``PRF_ORDER_PROJECT_THEN_CONVOLVE`` is False).

**Requirements**

- Default ``config.PRF_ORDER_PROJECT_THEN_CONVOLVE`` is **False** (convolve on scene, then
  project): identity patches on ``apply_spatially_varying_prf_*`` and bundle/exact paths are enough.
- If ``PRF_ORDER_PROJECT_THEN_CONVOLVE`` is **True**, this module also patches native PRF
  operators so **L_native ≡ I** and the forward remains projection-dominated.

Exact PRF mode (``PRF_OPERATOR_MODE`` resolving to ``exact``) uses a dense matrix **A**;
that path is overridden to **I** while this context is active.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Callable

import numpy as np

from . import solver


@contextmanager
def identity_prf_operators_context():
    """Replace scene and native PRF convolution with identity; exact mode uses **I**."""
    names = (
        "apply_spatially_varying_prf_to_scene",
        "apply_spatially_varying_prf_adjoint",
        "_get_prf_operator_bundle",
        "_apply_prf_operator_from_bundle",
        "_apply_prf_adjoint_from_bundle",
        "_get_prf_exact_operator_matrix",
        "_apply_prf_operator_native",
        "_apply_prf_adjoint_native",
    )
    orig: dict[str, Callable] = {n: getattr(solver, n) for n in names}

    def id_scene_fwd(intrinsic_scene, scene_wcs, w_native, scene_shape, channel, is_full_array=False):  # noqa: ARG001
        return np.asarray(intrinsic_scene, dtype=np.float64).reshape(scene_shape)

    def id_scene_adj(y_scene, scene_wcs, w_native, scene_shape, channel, is_full_array=False):  # noqa: ARG001
        y2 = np.asarray(y_scene, dtype=np.float64).reshape(scene_shape)
        return y2.ravel()

    def id_bundle(scene_wcs, w_native, scene_shape, channel, is_full_array=False):  # noqa: ARG001
        return (None, None, None)

    def id_from_bundle(img, kernels, weights, wsum):  # noqa: ARG001
        return np.asarray(img, dtype=np.float64)

    def id_adj_bundle(y, kernels, weights, wsum):  # noqa: ARG001
        return np.asarray(y, dtype=np.float64)

    def id_exact_matrix(scene_wcs, w_native, scene_shape, channel, is_full_array=False):  # noqa: ARG001
        h, w = int(scene_shape[0]), int(scene_shape[1])
        n = h * w
        return np.eye(n, dtype=np.float64)

    def id_native_fwd(img_native, native_wcs, native_shape, channel, is_full_array=False):  # noqa: ARG001
        return np.asarray(img_native, dtype=np.float64).reshape(native_shape)

    def id_native_adj(y_native, native_wcs, native_shape, channel, is_full_array=False):  # noqa: ARG001
        return np.asarray(y_native, dtype=np.float64).reshape(native_shape)

    try:
        solver.apply_spatially_varying_prf_to_scene = id_scene_fwd
        solver.apply_spatially_varying_prf_adjoint = id_scene_adj
        solver._get_prf_operator_bundle = id_bundle
        solver._apply_prf_operator_from_bundle = id_from_bundle
        solver._apply_prf_adjoint_from_bundle = id_adj_bundle
        solver._get_prf_exact_operator_matrix = id_exact_matrix
        solver._apply_prf_operator_native = id_native_fwd
        solver._apply_prf_adjoint_native = id_native_adj
        yield
    finally:
        for n in names:
            setattr(solver, n, orig[n])
