import os

import numpy as np
import pytest

from src import config
from src.pmap_correction import iracpc_pmap_corr


@pytest.mark.skipif(not os.path.isdir(config.PMAP_DIR), reason="PMAP_DIR not present")
def test_pmap_corr_finite_in_grid_ch1():
    flux_obs = 100.0
    x_test, y_test = 15.0, 15.0
    corr = iracpc_pmap_corr(
        flux_obs, x_test, y_test, 'ch1',
        pmap_dir=config.PMAP_DIR,
        threshold_occ=False,
    )
    assert np.isfinite(corr)
