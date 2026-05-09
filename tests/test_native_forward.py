import numpy as np
import pytest
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS

from src.native_forward import extract_native_stamp_for_target


def test_native_cutout_shape():
    w = WCS(naxis=2)
    w.wcs.crpix = [50.0, 50.0]
    w.wcs.crval = [197.45, -23.38]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.cdelt = [-0.0003, 0.0003]
    w.wcs.pc = np.eye(2)
    data = np.random.default_rng(0).random((100, 100)).astype(np.float64)
    unc = np.full_like(data, 0.01)
    target = SkyCoord(197.45, -23.38, unit="deg")
    d, s, wc = extract_native_stamp_for_target(data, unc, w, target, (32, 32))
    assert d.shape == (32, 32)
    assert s.shape == (32, 32)
    assert wc.naxis == 2
