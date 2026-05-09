import numpy as np

from src import config, solver


def test_predict_includes_host_gaussian_when_enabled(monkeypatch, tiny_cutouts):
    cutouts, wcs = tiny_cutouts
    monkeypatch.setattr(config, 'USE_HOST_GAUSSIAN_CORE', True)
    monkeypatch.setattr(config, 'HOST_CORE_RA', float(config.TRANSIENT_RA))
    monkeypatch.setattr(config, 'HOST_CORE_DEC', float(config.TRANSIENT_DEC))
    monkeypatch.setattr(config, 'HOST_CORE_SIGMA_PX', 2.0)

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
    results = dict(results)
    results['host_core_flux'] = 1e-4

    p0 = solver.predict_cutout_model(
        results, cutouts, [], [], 0,
        include_transient=False, include_stars=False, include_host=False,
    )
    p1 = solver.predict_cutout_model(
        results, cutouts, [], [], 0,
        include_transient=False, include_stars=False, include_host=True,
    )
    assert np.max(np.abs(p1 - p0)) > 1e-20


def test_predict_host_position_override_moves_kernel(tiny_cutouts):
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
    results = dict(results)
    results['host_core_flux'] = 2e-4
    ra0 = float(config.TRANSIENT_RA)
    dec0 = float(config.TRANSIENT_DEC)
    p0 = solver.predict_cutout_model(
        results, cutouts, [], [], 0,
        include_transient=False, include_stars=False, include_gp=False,
        include_host=True, host_position_override=(ra0, dec0),
    )
    p1 = solver.predict_cutout_model(
        results, cutouts, [], [], 0,
        include_transient=False, include_stars=False, include_gp=False,
        include_host=True, host_position_override=(ra0 + 2e-6, dec0),
    )
    assert np.max(np.abs(p1 - p0)) > 1e-24
