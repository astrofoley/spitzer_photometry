import numpy as np
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


def test_exact_prf_operator_matches_explicit_superposition(monkeypatch):
    monkeypatch.setattr(config, "PRF_OPERATOR_MODE", "exact")
    monkeypatch.setattr(config, "PRF_APPLY_PMAP_GAIN", False)
    solver._PRF_EXACT_OPERATOR_CACHE.clear()

    chan = str(getattr(config, "CHANNEL", "ch2"))
    ra0 = float(config.TRANSIENT_RA)
    dec0 = float(config.TRANSIENT_DEC)
    n = 10
    scene_shape = (n, n)
    scene_wcs = _scene_wcs(n, ra0, dec0, pixel_scale_arcsec=(config.PIXEL_SCALE / config.SUPERSAMPLE_FACTOR))
    native_wcs = _rot_wcs(n, ra0, dec0, theta_deg=19.0, pixel_scale_arcsec=config.PIXEL_SCALE)

    img = np.zeros(scene_shape, dtype=np.float64)
    img[2, 3] = 1.2
    img[5, 7] = 0.8
    img[8, 1] = 1.5

    via_operator = solver.apply_spatially_varying_prf_to_scene(
        img, scene_wcs, native_wcs, scene_shape, chan, is_full_array=True,
    )
    explicit = np.zeros(scene_shape, dtype=np.float64)
    ys, xs = np.nonzero(img)
    for y, x in zip(ys, xs):
        ra_p, dec_p = scene_wcs.pixel_to_world_values(float(x), float(y))
        col = solver.convolved_delta_column(
            scene_wcs, native_wcs, scene_shape, chan, float(ra_p), float(dec_p), is_full_array=True,
        ).reshape(scene_shape)
        explicit += float(img[y, x]) * col
    assert np.allclose(via_operator, explicit, atol=1e-10, rtol=1e-10)


def test_load_real_prf_file_preserves_shape_without_apodization(monkeypatch):
    raw = np.zeros((8, 8), dtype=np.float64)
    raw[2:6, 2:6] = 1.0

    monkeypatch.setattr(config, "PRF_APODIZATION_EDGE", 0.3)
    monkeypatch.setattr(config, "PRF_APPLY_APODIZATION", False)
    monkeypatch.setattr(solver, "_read_fits_image_array", lambda _: raw.copy())
    monkeypatch.setattr(solver.os.path, "exists", lambda _: True)

    loaded = solver.load_real_prf_file("ch2", 0, 0)
    assert np.array_equal(loaded, raw)


def test_generate_prf_fast_skips_pmap_when_disabled(monkeypatch):
    monkeypatch.setattr(config, "PRF_APPLY_PMAP_GAIN", False)
    calls = {"n": 0}

    def _fake_pmap(*args, **kwargs):
        calls["n"] += 1
        return 1.0

    monkeypatch.setattr(solver, "iracpc_pmap_corr", _fake_pmap)

    ra0 = float(config.TRANSIENT_RA)
    dec0 = float(config.TRANSIENT_DEC)
    n = 12
    scene_shape = (n, n)
    scene_wcs = _scene_wcs(n, ra0, dec0, pixel_scale_arcsec=(config.PIXEL_SCALE / config.SUPERSAMPLE_FACTOR))
    native_wcs = _rot_wcs(n, ra0, dec0, theta_deg=0.0, pixel_scale_arcsec=config.PIXEL_SCALE)
    prf = np.zeros((31, 31), dtype=np.float64)
    prf[15, 15] = 1.0

    out = solver.generate_prf_fast(scene_wcs, native_wcs, prf, ra0, dec0, scene_shape, channel="ch2", is_full_array=True)
    assert np.isfinite(out).all()
    assert calls["n"] == 0
