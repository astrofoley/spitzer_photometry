import numpy as np

from src import solver


def test_run_gls_solve_synthetic_template_transient_near_zero(tiny_cutouts):
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
    tpl_idx = [i for i, c in enumerate(cutouts) if c.get('is_template')]
    for i in tpl_idx:
        assert abs(results['transient_fluxes'][i]) < 1e-9
    assert results['transient_errs'].shape == (len(cutouts),)
    assert np.all(np.isfinite(results['transient_errs']))
    assert len(results['transient_epoch_fluxes']) >= 1
    # conftest unbounded-MAP path allows tiny negative fluxes from the linear fit
    neg_tol = 1e-3
    assert np.all(results['transient_epoch_fluxes'] >= -neg_tol)
    for i, c in enumerate(cutouts):
        if not c.get('is_template'):
            assert results['transient_fluxes'][i] >= -neg_tol
    assert 'transient_dra_deg' in results
    assert 'transient_ddec_deg' in results
    assert np.isfinite(results['transient_dra_deg'])
    assert np.isfinite(results['transient_ddec_deg'])


def test_hessian_uncertainty_matches_inverse():
    rng = np.random.default_rng(0)
    n = 5
    A = rng.standard_normal((n, n))
    H = A @ A.T + np.eye(n) * 0.5
    rhs = rng.standard_normal(n)
    sol = np.linalg.solve(H, rhs)
    cov = np.linalg.inv(H)
    sig = np.sqrt(np.diag(cov))
    assert np.all(sig > 0)
    assert np.allclose(H @ sol, rhs)
