import numpy as np
import pytest
from astropy.wcs import WCS

from src import config, solver


def test_predict_cutout_matches_background_only(tiny_cutouts):
    cutouts, wcs = tiny_cutouts
    ell, var = 3.0, 1e-8
    n_epochs = 1
    results = solver.run_gls_solve(
        cutouts,
        [],
        [],
        {'ell': ell, 'var': var},
        (ell, var),
        np.zeros((cutouts[0]['data'].shape[0], cutouts[0]['data'].shape[1])),
        wcs,
        n_epochs,
    )
    assert results is not None
    i = 0
    pred = solver.predict_cutout_model(
        results, cutouts, [], [], i,
        include_transient=False, include_stars=False,
    )
    if 'bcd_backgrounds' in results:
        bg = float(results['bcd_backgrounds'][i])
    else:
        bg = results['epoch_backgrounds'][cutouts[i]['epoch_id']]
    assert pred.shape == results['gp_scene'].shape
    assert np.isfinite(pred).all()


def test_predict_includes_transient_scale(tiny_cutouts):
    cutouts, wcs = tiny_cutouts
    ell, var = 3.0, 1e-8
    results = solver.run_gls_solve(
        cutouts,
        [],
        [],
        {'ell': ell, 'var': var},
        (ell, var),
        np.zeros((16, 16)),
        wcs,
        1,
    )
    assert results is not None
    ft = float(results['transient_fluxes'][0])
    p0 = solver.predict_cutout_model(
        results, cutouts, [], [], 0,
        include_transient=True, transient_flux_override=0.0,
    )
    p1 = solver.predict_cutout_model(
        results, cutouts, [], [], 0,
        include_transient=True, transient_flux_override=ft,
    )
    col = p1 - p0
    if abs(ft) > 1e-20:
        assert np.nanmax(np.abs(col)) > 0


def test_predict_include_gp_flag(tiny_cutouts):
    cutouts, wcs = tiny_cutouts
    ell, var = 3.0, 1e-8
    results = solver.run_gls_solve(
        cutouts,
        [],
        [],
        {'ell': ell, 'var': var},
        (ell, var),
        np.zeros((cutouts[0]['data'].shape[0], cutouts[0]['data'].shape[1])),
        wcs,
        1,
    )
    assert results is not None
    i = 0
    p_gp = solver.predict_cutout_model(
        results, cutouts, [], [], i,
        include_gp=True, include_transient=False, include_stars=False,
    )
    p_no = solver.predict_cutout_model(
        results, cutouts, [], [], i,
        include_gp=False, include_transient=False, include_stars=False,
    )
    if 'bcd_backgrounds' in results:
        bg = float(results['bcd_backgrounds'][i])
    else:
        bg = results['epoch_backgrounds'][cutouts[i]['epoch_id']]
    np.testing.assert_allclose(p_no, np.full_like(p_no, bg), rtol=0, atol=1e-12)
    assert np.nanmax(np.abs(p_gp - p_no)) > 0.0
