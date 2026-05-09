"""Tests for diagnostic identity PRF context."""
import numpy as np

from src import solver
from src.prf_identity_context import identity_prf_operators_context


def test_identity_prf_scene_forward_preserves_array():
    h, w = 5, 7
    scene_shape = (h, w)
    intrinsic = np.arange(h * w, dtype=np.float64).reshape(scene_shape)
    with identity_prf_operators_context():
        out = solver.apply_spatially_varying_prf_to_scene(
            intrinsic, None, None, scene_shape, "ch2",
        )
    assert out.shape == scene_shape
    np.testing.assert_allclose(out, intrinsic)


def test_identity_prf_exact_matrix_is_identity():
    from astropy.wcs import WCS

    h, w = 4, 3
    scene_shape = (h, w)
    sw = WCS(naxis=2)
    sw.wcs.crpix = [2.0, 2.0]
    sw.wcs.crval = [10.0, 20.0]
    sw.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    sw.wcs.cdelt = [-0.001, 0.001]
    sw.wcs.pc = np.eye(2)
    nw = sw.deepcopy()
    n = h * w
    with identity_prf_operators_context():
        a = solver._get_prf_exact_operator_matrix(
            sw, nw, scene_shape, "ch2", is_full_array=False,
        )
    assert a.shape == (n, n)
    np.testing.assert_allclose(a, np.eye(n))


def test_identity_prf_native_forward_preserves():
    from astropy.wcs import WCS

    native_shape = (6, 8)
    img = np.ones(native_shape, dtype=np.float64)
    nw = WCS(naxis=2)
    nw.wcs.crpix = [4.0, 3.0]
    nw.wcs.crval = [10.0, 20.0]
    nw.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    nw.wcs.cdelt = [-0.001, 0.001]
    nw.wcs.pc = np.eye(2)
    with identity_prf_operators_context():
        out = solver._apply_prf_operator_native(img, nw, native_shape, "ch2", False)
    np.testing.assert_allclose(out, img)
