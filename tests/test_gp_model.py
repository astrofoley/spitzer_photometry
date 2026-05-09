import numpy as np
import pytest

from src import config
from src import gp_model


def test_build_scene_prior_inverse_full_matern_shape():
    h, w = 8, 8
    n = h * w
    Qinv = gp_model.build_scene_prior_inverse(n, ell=2.0, var=0.5, scene_shape=(h, w))
    assert Qinv.shape == (n, n)
    assert np.all(np.isfinite(Qinv))


def test_build_scene_prior_diagonal_fallback(monkeypatch):
    monkeypatch.setattr(config, 'MAX_SCENE_PIXELS', 10)
    monkeypatch.setattr(config, 'GP_FALLBACK_NEIGHBOR_SMOOTHNESS', 0.0)
    h, w = 4, 4
    n = h * w
    with pytest.warns(UserWarning, match="exceeds MAX_SCENE_PIXELS"):
        Qinv = gp_model.build_scene_prior_inverse(n, ell=2.0, var=0.25, scene_shape=(h, w))
    assert Qinv.shape == (n, n)
    assert np.allclose(Qinv, np.diag(np.diag(Qinv)))


def test_build_scene_prior_smoothed_fallback(monkeypatch):
    monkeypatch.setattr(config, 'MAX_SCENE_PIXELS', 10)
    monkeypatch.setattr(config, 'GP_FALLBACK_NEIGHBOR_SMOOTHNESS', 0.2)
    h, w = 4, 4
    n = h * w
    with pytest.warns(UserWarning, match="exceeds MAX_SCENE_PIXELS"):
        Qinv = gp_model.build_scene_prior_inverse(n, ell=2.0, var=0.25, scene_shape=(h, w))
    assert Qinv.shape == (n, n)
    # Smoothed fallback should include off-diagonal neighbor couplings.
    assert not np.allclose(Qinv, np.diag(np.diag(Qinv)))


def test_matern32_kernel_diag_positive():
    coords = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    K = gp_model.matern32_kernel(coords, length_scale=2.0, variance=1.0)
    assert K.shape == (3, 3)
    assert np.all(np.diag(K) > 0)


def test_matern12_kernel_diag_positive():
    coords = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    K = gp_model.matern12_kernel(coords, length_scale=2.0, variance=1.0)
    assert K.shape == (3, 3)
    assert np.all(np.diag(K) > 0)


def test_build_scene_prior_inverse_matern12(monkeypatch):
    monkeypatch.setattr(config, "GP_MATERN_ORDER", "matern12")
    h, w = 8, 8
    n = h * w
    Qinv = gp_model.build_scene_prior_inverse(n, ell=2.0, var=0.5, scene_shape=(h, w))
    assert Qinv.shape == (n, n)
    assert np.all(np.isfinite(Qinv))
    assert np.allclose(Qinv, Qinv.T)


def test_build_scene_prior_diagonal_fallback_strict_zero_neighbor(monkeypatch):
    """Ablation mode: MAX_SCENE_PIXELS=0 and zero neighbor smoothness => strictly diagonal Qinv."""
    monkeypatch.setattr(config, "MAX_SCENE_PIXELS", 0)
    monkeypatch.setattr(config, "GP_FALLBACK_NEIGHBOR_SMOOTHNESS", 0.0)
    h, w = 4, 4
    n = h * w
    with pytest.warns(UserWarning, match="exceeds MAX_SCENE_PIXELS"):
        Qinv = gp_model.build_scene_prior_inverse(n, ell=2.0, var=0.25, scene_shape=(h, w))
    assert Qinv.shape == (n, n)
    assert np.allclose(Qinv, np.diag(np.diag(Qinv)))
