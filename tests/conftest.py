import numpy as np
import pytest
from astropy.wcs import WCS

from src import config


@pytest.fixture(autouse=True)
def _fast_joint_solver_tests():
    """
    Default pipeline settings (3×3 PRF anchors, transient subpixel solve, bounded MAP) make
    matrix assembly and trust-constr too slow for routine unit tests. Restore after each test.
    """
    saved = (
        config.PRF_SPATIAL_ANCHORS_PER_AXIS,
        config.FLOAT_TRANSIENT_POSITION,
        config.TRANSIENT_NONNEGATIVE,
    )
    config.PRF_SPATIAL_ANCHORS_PER_AXIS = 1
    config.FLOAT_TRANSIENT_POSITION = False
    config.TRANSIENT_NONNEGATIVE = False
    yield
    config.PRF_SPATIAL_ANCHORS_PER_AXIS = saved[0]
    config.FLOAT_TRANSIENT_POSITION = saved[1]
    config.TRANSIENT_NONNEGATIVE = saved[2]


@pytest.fixture
def tiny_cutouts():
    n_pix = 16
    n_frames = 3
    target_ra = float(config.TRANSIENT_RA)
    target_dec = float(config.TRANSIENT_DEC)
    w_small = WCS(naxis=2)
    w_small.wcs.crpix = [n_pix / 2, n_pix / 2]
    w_small.wcs.crval = [target_ra, target_dec]
    w_small.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    scale = config.PIXEL_SCALE / config.SUPERSAMPLE_FACTOR / 3600.0
    w_small.wcs.cdelt = [-scale, scale]
    w_small.wcs.pc = np.eye(2)
    w_native = w_small.deepcopy()
    cutouts = []
    rng = np.random.default_rng(42)
    for i in range(n_frames):
        d = rng.normal(1e-4, 1e-5, (n_pix, n_pix)).astype(np.float64)
        s = np.full_like(d, 1e-5)
        cutouts.append({
            'data': d,
            'sigma': s,
            'wcs': w_small,
            'raw_wcs': w_native,
            'is_full_array': True,
            'mjd': 58000.0 + i,
            'filename': f'synthetic_ch2_{i:03d}_cbcd.fits',
            'epoch_id': 0,
            'is_template': (i == n_frames - 1),
        })
    return cutouts, w_small
