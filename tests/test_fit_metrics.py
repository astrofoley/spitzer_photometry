import numpy as np
from astropy.wcs import WCS

from src import fit_metrics


def _toy_wcs(n_pix=10, ra=1.0, dec=2.0):
    w = WCS(naxis=2)
    w.wcs.crpix = [n_pix / 2, n_pix / 2]
    w.wcs.crval = [ra, dec]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.cdelt = [-1.0 / 3600.0, 1.0 / 3600.0]
    w.wcs.pc = np.eye(2)
    return w


def test_compute_fit_metrics_basic(monkeypatch):
    # Monkeypatch model predictor to return zeros.
    from src import solver

    monkeypatch.setattr(
        solver,
        "predict_cutout_model",
        lambda *args, **kwargs: np.zeros((10, 10), dtype=float),
    )
    w = _toy_wcs()
    cutouts = [
        {
            "data": np.ones((10, 10), dtype=float),
            "sigma": np.ones((10, 10), dtype=float),
            "wcs": w,
            "raw_wcs": w,
            "is_template": True,
            "epoch_id": 0,
            "filename": "x",
        }
    ]
    res = fit_metrics.compute_fit_metrics(
        cutouts,
        results={},
        stars=[],
        star_fluxes=[],
        center_ra_deg=1.0,
        center_dec_deg=2.0,
        center_radius_px=2.0,
    )
    assert res["total_ndof"] > 0
    assert res["center_ndof"] > 0
    assert np.isfinite(res["center_reduced_chi2"])
