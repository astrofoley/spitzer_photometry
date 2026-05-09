import numpy as np

from src import solver


def test_trim_prf_zero_padding_removes_outer_zeros():
    prf = np.zeros((8, 10), dtype=np.float64)
    prf[2:6, 3:9] = 1.0
    trimmed = solver.trim_prf_zero_padding(prf)
    assert trimmed.shape == (4, 6)
    assert np.all(trimmed == 1.0)


def test_load_prf_has_nonzero_native_outer_ring():
    # After trimming file padding, at least one value on the outer 1-px ring
    # should be nonzero for real SSC PRFs.
    prf = np.asarray(solver.load_prf('ch2', 128.0, 128.0), dtype=np.float64)
    assert prf.ndim == 2
    assert prf.shape[0] > 2 and prf.shape[1] > 2
    s = float(np.sum(prf))
    assert s > 0.0
    p = prf / s
    border = np.zeros_like(p, dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    assert float(np.max(p[border])) > 0.0
