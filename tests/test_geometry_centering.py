import numpy as np
import pytest
from astropy.wcs import WCS

from src import config, solver


def _scene_wcs(n_pix: int, ra: float, dec: float, pixel_scale_arcsec: float) -> WCS:
    w = WCS(naxis=2)
    w.wcs.crpix = [n_pix / 2, n_pix / 2]
    w.wcs.crval = [ra, dec]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    scale = float(pixel_scale_arcsec) / 3600.0
    w.wcs.cdelt = [-scale, scale]
    w.wcs.pc = np.eye(2)
    return w


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


def _centroid_xy(img: np.ndarray) -> tuple[float, float]:
    a = np.asarray(img, dtype=float)
    a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
    a[a < 0.0] = 0.0
    s = float(np.sum(a))
    if s <= 0.0:
        iy, ix = np.unravel_index(int(np.argmax(np.abs(img))), img.shape)
        return float(ix), float(iy)
    yy, xx = np.mgrid[0 : a.shape[0], 0 : a.shape[1]].astype(float)
    cx = float(np.sum(xx * a) / s)
    cy = float(np.sum(yy * a) / s)
    return cx, cy


@pytest.mark.parametrize("theta_deg", [0.0, 17.0, 33.0])
def test_projection_preserves_delta_center_to_subpixel(theta_deg: float):
    """Projection alone should keep a source near its expected native sky position."""
    ra0 = float(config.TRANSIENT_RA)
    dec0 = float(config.TRANSIENT_DEC)
    n_scene = 40
    n_native = 40
    scene_wcs = _scene_wcs(n_scene, ra0, dec0, pixel_scale_arcsec=(config.PIXEL_SCALE / config.SUPERSAMPLE_FACTOR))
    native_wcs = _rot_wcs(n_native, ra0, dec0, theta_deg=float(theta_deg), pixel_scale_arcsec=config.PIXEL_SCALE)
    scene_shape = (n_scene, n_scene)
    native_shape = (n_native, n_native)

    # Choose an off-center source to exercise interpolation phase.
    xs = float(n_scene / 2 + 3.25)
    ys = float(n_scene / 2 - 4.5)
    ra_s, dec_s = scene_wcs.pixel_to_world_values(xs, ys)

    scene = np.zeros(scene_shape, dtype=float)
    solver._add_delta_to_image(scene, xs, ys, 1.0)
    native = solver._project_scene_to_native(scene, scene_wcs, native_wcs, native_shape)
    cx, cy = _centroid_xy(native)
    xexp, yexp = native_wcs.world_to_pixel_values(float(ra_s), float(dec_s))
    dr = float(np.hypot(cx - float(xexp), cy - float(yexp)))
    assert dr < 0.35, f"Projection center drift too large ({dr:.3f} px) at theta={theta_deg}"


@pytest.mark.parametrize("theta_deg", [0.0, 23.0])
def test_frame_forward_orders_match_when_prf_is_identity(monkeypatch, theta_deg: float):
    """
    If PRF is identity, both forward orders should reduce to the same projection map.
    This isolates order-switch geometry wiring.
    """
    ra0 = float(config.TRANSIENT_RA)
    dec0 = float(config.TRANSIENT_DEC)
    n_scene = 40
    n_native = 40
    scene_wcs = _scene_wcs(n_scene, ra0, dec0, pixel_scale_arcsec=(config.PIXEL_SCALE / config.SUPERSAMPLE_FACTOR))
    native_wcs = _rot_wcs(n_native, ra0, dec0, theta_deg=float(theta_deg), pixel_scale_arcsec=config.PIXEL_SCALE)
    scene_shape = (n_scene, n_scene)
    native_shape = (n_native, n_native)

    def _id_scene(scene_img, *_args, **_kwargs):
        return np.asarray(scene_img, dtype=float)

    def _id_native(native_img, *_args, **_kwargs):
        return np.asarray(native_img, dtype=float)

    monkeypatch.setattr(solver, "apply_spatially_varying_prf_to_scene", _id_scene)
    monkeypatch.setattr(solver, "_apply_prf_operator_native", _id_native)

    x = np.zeros(scene_shape, dtype=float)
    solver._add_delta_to_image(x, n_scene / 2 + 2.4, n_scene / 2 - 1.7, 1.0)

    with nfc_temporary_config({"PRF_ORDER_PROJECT_THEN_CONVOLVE": False}):
        old_order = solver._apply_frame_forward_operator(
            x, scene_wcs, native_wcs, scene_shape, native_shape, str(config.CHANNEL), is_full_array=True
        )
    with nfc_temporary_config({"PRF_ORDER_PROJECT_THEN_CONVOLVE": True}):
        new_order = solver._apply_frame_forward_operator(
            x, scene_wcs, native_wcs, scene_shape, native_shape, str(config.CHANNEL), is_full_array=True
        )

    diff = float(np.max(np.abs(np.asarray(old_order) - np.asarray(new_order))))
    assert diff < 1e-9, f"Order switch changes projection even with identity PRF (max abs diff {diff:.3e})"


class nfc_temporary_config:
    """Local lightweight context manager to avoid importing campaign module in tests."""

    def __init__(self, overrides: dict):
        self.overrides = dict(overrides)
        self.old = {}

    def __enter__(self):
        for k, v in self.overrides.items():
            self.old[k] = getattr(config, k)
            setattr(config, k, v)
        return self

    def __exit__(self, exc_type, exc, tb):
        for k, v in self.old.items():
            setattr(config, k, v)
        return False
