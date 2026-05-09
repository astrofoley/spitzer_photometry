import numpy as np
import pytest
from astropy.wcs import WCS

from src import config, solver


def _rot_wcs(n_pix: int, ra: float, dec: float, theta_deg: float, pixel_scale_arcsec: float) -> WCS:
    w = WCS(naxis=2)
    w.wcs.crpix = [n_pix / 2, n_pix / 2]
    w.wcs.crval = [ra, dec]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    scale = float(pixel_scale_arcsec) / 3600.0
    w.wcs.cdelt = [-scale, scale]
    th = np.deg2rad(theta_deg)
    c, s = np.cos(th), np.sin(th)
    w.wcs.pc = np.array([[c, -s], [s, c]])
    return w


def _scene_wcs(n_pix: int, ra: float, dec: float, pixel_scale_arcsec: float) -> WCS:
    w = WCS(naxis=2)
    w.wcs.crpix = [n_pix / 2, n_pix / 2]
    w.wcs.crval = [ra, dec]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    scale = float(pixel_scale_arcsec) / 3600.0
    w.wcs.cdelt = [-scale, scale]
    w.wcs.pc = np.eye(2)
    return w


@pytest.mark.parametrize("theta_deg", [0.0, 17.0])
def test_frame_operator_adjoint_inner_product(theta_deg: float):
    """
    Fundamental math check:
      <F x, y> == <x, F^T y>

    Our adjoint is approximate (reprojection + spatially varying PRF), so allow
    a small tolerance. If this fails badly, extended-mode fitting can break even
    when point-like columns appear plausible.
    """
    rng = np.random.default_rng(0)
    chan = str(getattr(config, "CHANNEL", "ch2"))
    ra0 = float(config.TRANSIENT_RA)
    dec0 = float(config.TRANSIENT_DEC)
    n_scene = 32
    n_native = 32
    scene_wcs = _scene_wcs(n_scene, ra0, dec0, pixel_scale_arcsec=(config.PIXEL_SCALE / config.SUPERSAMPLE_FACTOR))
    native_wcs = _rot_wcs(n_native, ra0, dec0, theta_deg=float(theta_deg), pixel_scale_arcsec=config.PIXEL_SCALE)

    scene_shape = (n_scene, n_scene)
    native_shape = (n_native, n_native)

    x = rng.normal(size=scene_shape).astype(np.float64)
    y = rng.normal(size=native_shape).astype(np.float64)

    Fx = solver._apply_frame_forward_operator(
        x,
        scene_wcs,
        native_wcs,
        scene_shape,
        native_shape,
        chan,
        is_full_array=True,
    )
    FTy = solver._apply_frame_adjoint_operator(
        y,
        scene_wcs,
        native_wcs,
        scene_shape,
        chan,
        is_full_array=True,
    )

    lhs = float(np.vdot(Fx, y))
    rhs = float(np.vdot(x, FTy))
    denom = max(1e-12, abs(lhs), abs(rhs))
    rel = abs(lhs - rhs) / denom
    assert rel < 5e-2, f"Adjoint inner-product mismatch too large: rel={rel:.3g} lhs={lhs:.6g} rhs={rhs:.6g}"


@pytest.mark.parametrize("theta_deg", [0.0, 17.0])
def test_projection_operator_adjoint_inner_product(theta_deg: float):
    """
    Localize mismatch: projection only.
      <P x, y> == <x, P^T y>
    """
    rng = np.random.default_rng(1)
    ra0 = float(config.TRANSIENT_RA)
    dec0 = float(config.TRANSIENT_DEC)
    n_scene = 32
    n_native = 32
    scene_wcs = _scene_wcs(n_scene, ra0, dec0, pixel_scale_arcsec=(config.PIXEL_SCALE / config.SUPERSAMPLE_FACTOR))
    native_wcs = _rot_wcs(n_native, ra0, dec0, theta_deg=float(theta_deg), pixel_scale_arcsec=config.PIXEL_SCALE)
    scene_shape = (n_scene, n_scene)
    native_shape = (n_native, n_native)

    x = rng.normal(size=scene_shape).astype(np.float64)
    y = rng.normal(size=native_shape).astype(np.float64)
    Px = solver._project_scene_to_native(x, scene_wcs, native_wcs, native_shape)
    PTy = solver._project_native_to_scene(y, native_wcs, scene_wcs, scene_shape)
    lhs = float(np.vdot(Px, y))
    rhs = float(np.vdot(x, PTy))
    denom = max(1e-12, abs(lhs), abs(rhs))
    rel = abs(lhs - rhs) / denom
    assert rel < 5e-2, f"Projection adjoint mismatch: rel={rel:.3g} lhs={lhs:.6g} rhs={rhs:.6g}"


def test_frame_operator_constant_scene_reasonable():
    """
    Sanity check: constant scene should not explode in native prediction.
    This catches gross scaling/normalization mistakes.
    """
    chan = str(getattr(config, "CHANNEL", "ch2"))
    ra0 = float(config.TRANSIENT_RA)
    dec0 = float(config.TRANSIENT_DEC)
    n_scene = 32
    n_native = 32
    scene_wcs = _scene_wcs(n_scene, ra0, dec0, pixel_scale_arcsec=(config.PIXEL_SCALE / config.SUPERSAMPLE_FACTOR))
    native_wcs = _rot_wcs(n_native, ra0, dec0, theta_deg=13.0, pixel_scale_arcsec=config.PIXEL_SCALE)
    scene_shape = (n_scene, n_scene)
    native_shape = (n_native, n_native)

    x = np.ones(scene_shape, dtype=np.float64)
    Fx = solver._apply_frame_forward_operator(
        x,
        scene_wcs,
        native_wcs,
        scene_shape,
        native_shape,
        chan,
        is_full_array=True,
    )
    assert np.isfinite(Fx).all()
    # Should remain O(1) for unit scene (not blow up to huge magnitudes).
    assert float(np.nanmax(np.abs(Fx))) < 1e3


@pytest.mark.parametrize("theta_deg", [0.0, 17.0])
def test_prf_operator_adjoint_inner_product(theta_deg: float):
    """
    Localize mismatch: PRF operator only (L vs L^T) on the scene grid.
      <L x, y> == <x, L^T y>
    """
    rng = np.random.default_rng(2)
    chan = str(getattr(config, "CHANNEL", "ch2"))
    ra0 = float(config.TRANSIENT_RA)
    dec0 = float(config.TRANSIENT_DEC)
    n_scene = 32
    n_native = 32
    scene_wcs = _scene_wcs(n_scene, ra0, dec0, pixel_scale_arcsec=(config.PIXEL_SCALE / config.SUPERSAMPLE_FACTOR))
    native_wcs = _rot_wcs(n_native, ra0, dec0, theta_deg=float(theta_deg), pixel_scale_arcsec=config.PIXEL_SCALE)
    scene_shape = (n_scene, n_scene)

    kernels, weights, wsum = solver._get_prf_operator_bundle(
        scene_wcs, native_wcs, scene_shape, chan, True,
    )
    x = rng.normal(size=scene_shape).astype(np.float64)
    y = rng.normal(size=scene_shape).astype(np.float64)
    Lx = solver._apply_prf_operator_from_bundle(x, kernels, weights, wsum)
    LTy = solver._apply_prf_adjoint_from_bundle(y, kernels, weights, wsum)
    lhs = float(np.vdot(Lx, y))
    rhs = float(np.vdot(x, LTy))
    denom = max(1e-12, abs(lhs), abs(rhs))
    rel = abs(lhs - rhs) / denom
    assert rel < 5e-2, f"PRF adjoint mismatch: rel={rel:.3g} lhs={lhs:.6g} rhs={rhs:.6g}"
