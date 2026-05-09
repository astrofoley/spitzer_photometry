import numpy as np

from src import residual_metrics


def test_summarize_white_noise_small_lag1():
    rng = np.random.default_rng(0)
    n = 32
    resid = rng.normal(0.0, 1.0, (n, n))
    sigma = np.ones_like(resid)
    mask = np.ones_like(resid, dtype=bool)
    out = residual_metrics.summarize_frame_residual(resid, sigma, mask)
    assert 'acf_e90_scale_pix' in out
    assert 'z_lag1_corr' in out
    assert abs(float(out['z_lag1_corr'])) < 0.25


def test_dipole_detects_offset_blob():
    h, w = 24, 24
    yy, xx = np.mgrid[0:h, 0:w]
    blob = np.exp(-((xx - 8.0) ** 2 + (yy - 12.0) ** 2) / (2 * 2.0 ** 2))
    mx, my, mag = residual_metrics.dipole_moment_xy(blob, np.ones_like(blob, dtype=bool))
    assert mx < -0.5
    assert abs(my) < 1.0
    assert mag > 0.5
