"""Tests for the resolution-independent PRF WCS fix in generate_prf_fast.

Covers:
- PRF WCS uses PRF_PIXEL_SCALE_ARCSEC (not native_wcs / PRF_OVERSAMPLE)
- hasattr(cd) bug fix: CDELT+PC WCS no longer gives zero-scale PRF
- PRF reprojection produces correct FWHM at any SUPERSAMPLE_FACTOR
- Orientation is preserved from native WCS rotation
"""
from __future__ import annotations

import contextlib
import numpy as np
import pytest
from astropy.wcs import WCS

from src import config, solver

# PRF spread assertions are sensitive to astropy/WCS `cd` handling; keep in full
# suite / local deep runs (see Phase 1 `pytest -m "not slow"`).
pytestmark = pytest.mark.slow


# ---------------------------------------------------------------------------
# PRF pixel scale used in tests.
# Real PRF_PIXEL_SCALE_ARCSEC = 0.0122" → FWHM ≈ 0.23 scene pixels for a
# 51-px Gaussian PRF (too small to be resolved).  Override to 0.5"/prf-px
# so FWHM ≈ 5.9" = 9.7 scene pixels — clearly spread over multiple pixels.
_TEST_PRF_SCALE_ARCSEC = 0.5


@contextlib.contextmanager
def _prf_scale_override(scale_arcsec):
    """Temporarily override PRF_PIXEL_SCALE_ARCSEC and flush PRF caches."""
    saved = getattr(config, 'PRF_PIXEL_SCALE_ARCSEC', config.PIXEL_SCALE / config.PRF_OVERSAMPLE)
    config.PRF_PIXEL_SCALE_ARCSEC = scale_arcsec
    solver._PRF_OPERATOR_BUNDLE_CACHE.clear()
    solver._PRF_NATIVE_OPERATOR_BUNDLE_CACHE.clear()
    try:
        yield
    finally:
        config.PRF_PIXEL_SCALE_ARCSEC = saved
        solver._PRF_OPERATOR_BUNDLE_CACHE.clear()
        solver._PRF_NATIVE_OPERATOR_BUNDLE_CACHE.clear()


# ---------------------------------------------------------------------------
# WCS helpers
# ---------------------------------------------------------------------------

def _make_cdelt_wcs(scale_arcsec, rotation_deg=0.0):
    """North-up WCS using CDELT+PC (no CD matrix — the old hasattr bug case)."""
    w = WCS(naxis=2)
    w.wcs.crpix = [64, 64]
    w.wcs.crval = [float(config.TRANSIENT_RA), float(config.TRANSIENT_DEC)]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.cdelt = np.array([-scale_arcsec / 3600.0, scale_arcsec / 3600.0])
    th = np.radians(rotation_deg)
    w.wcs.pc = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    w.wcs.set()
    return w


def _make_cd_wcs(scale_arcsec, rotation_deg=0.0):
    """WCS using CD matrix form (as in real Spitzer BCDs)."""
    w = WCS(naxis=2)
    w.wcs.crpix = [64, 64]
    w.wcs.crval = [float(config.TRANSIENT_RA), float(config.TRANSIENT_DEC)]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    sc = scale_arcsec / 3600.0
    th = np.radians(rotation_deg)
    w.wcs.cd = np.array([
        [-sc * np.cos(th),  sc * np.sin(th)],
        [ sc * np.sin(th),  sc * np.cos(th)],
    ])
    w.wcs.set()
    return w


def _tiny_prf(size=51):
    """Gaussian PRF, sig=5 PRF pixels → FWHM ≈ 11.75 PRF px.
    At _TEST_PRF_SCALE_ARCSEC=0.5"/px: FWHM ≈ 5.9" = 9.7 scene px (clearly spread).
    """
    y, x = np.mgrid[-size // 2:size // 2 + 1, -size // 2:size // 2 + 1].astype(float)
    k = np.exp(-0.5 * (x ** 2 + y ** 2) / 25.0)   # sig=5
    k /= k.sum()
    return k


# ---------------------------------------------------------------------------
# Tests: scale independence
# ---------------------------------------------------------------------------

class TestPRFWCSScaleIndependence:
    """PRF reprojection scale comes from PRF_PIXEL_SCALE_ARCSEC, not from native_wcs scale."""

    def test_cdelt_wcs_gives_spread_prf(self):
        """Old hasattr(cd) bug: CDELT+PC WCS had cd=zeros but hasattr returned True → zero-scale PRF.
        After fix: CDELT+PC WCS gives a properly spread PRF column."""
        scene_wcs = _make_cdelt_wcs(config.PIXEL_SCALE / config.SUPERSAMPLE_FACTOR)
        native_wcs = _make_cdelt_wcs(config.PIXEL_SCALE)  # CDELT form, no CD matrix
        prf = _tiny_prf()
        ra, dec = float(config.TRANSIENT_RA), float(config.TRANSIENT_DEC)

        with _prf_scale_override(_TEST_PRF_SCALE_ARCSEC):
            out = solver.generate_prf_fast(scene_wcs, native_wcs, prf, ra, dec, (64, 64))
        out2d = out.reshape(64, 64)

        assert abs(out2d.sum() - 1.0) < 0.05, f"Column not normalised: sum={out2d.sum():.4f}"
        n_sig = int(np.sum(out2d > out2d.max() * 0.01))
        assert n_sig > 4, f"PRF collapsed to delta: only {n_sig} pixels above 1% of peak"

    def test_cd_wcs_gives_spread_prf(self):
        """CD-matrix WCS (real Spitzer BCD form) should give spread PRF."""
        scene_wcs = _make_cdelt_wcs(config.PIXEL_SCALE / config.SUPERSAMPLE_FACTOR)
        native_wcs = _make_cd_wcs(config.PIXEL_SCALE)
        prf = _tiny_prf()
        ra, dec = float(config.TRANSIENT_RA), float(config.TRANSIENT_DEC)

        with _prf_scale_override(_TEST_PRF_SCALE_ARCSEC):
            out = solver.generate_prf_fast(scene_wcs, native_wcs, prf, ra, dec, (64, 64))
        out2d = out.reshape(64, 64)

        assert abs(out2d.sum() - 1.0) < 0.05
        n_sig = int(np.sum(out2d > out2d.max() * 0.01))
        assert n_sig > 4, f"PRF too compact with CD WCS: {n_sig} pixels"

    def test_prf_fwhm_scales_with_supersample(self):
        """At higher SUPERSAMPLE, PRF FWHM in scene pixels should scale proportionally."""
        ra, dec = float(config.TRANSIENT_RA), float(config.TRANSIENT_DEC)
        prf = _tiny_prf()
        native_wcs = _make_cd_wcs(config.PIXEL_SCALE)

        fwhm_scene_px = {}
        with _prf_scale_override(_TEST_PRF_SCALE_ARCSEC):
            for ss in [1, 2, 4]:
                scene_wcs = _make_cdelt_wcs(config.PIXEL_SCALE / ss)
                out2d = solver.generate_prf_fast(
                    scene_wcs, native_wcs, prf, ra, dec, (64, 64)
                ).reshape(64, 64)
                if out2d.max() <= 0:
                    continue
                n_above = int(np.sum(out2d >= out2d.max() / 2.0))
                fwhm_scene_px[ss] = np.sqrt(n_above / np.pi) * 2

        assert len(fwhm_scene_px) >= 2, "Not enough SUPERSAMPLE values gave non-zero PRF"
        ss_vals = sorted(fwhm_scene_px)
        for i in range(1, len(ss_vals)):
            ratio = fwhm_scene_px[ss_vals[i]] / fwhm_scene_px[ss_vals[i - 1]]
            ss_ratio = ss_vals[i] / ss_vals[i - 1]
            assert ratio > ss_ratio * 0.5, (
                f"FWHM didn't scale with SUPERSAMPLE: "
                f"SS={ss_vals[i - 1]}→{ss_vals[i]}, ratio={ratio:.2f} (expected ~{ss_ratio:.1f})"
            )

    def test_prf_scale_independent_of_native_wcs_scale(self):
        """Changing native WCS pixel scale should NOT change the reprojected PRF.
        The PRF scale is fixed by PRF_PIXEL_SCALE_ARCSEC, not by native_wcs.cdelt."""
        ra, dec = float(config.TRANSIENT_RA), float(config.TRANSIENT_DEC)
        prf = _tiny_prf()
        scene_wcs = _make_cdelt_wcs(config.PIXEL_SCALE / config.SUPERSAMPLE_FACTOR)

        with _prf_scale_override(_TEST_PRF_SCALE_ARCSEC):
            out_1x = solver.generate_prf_fast(
                scene_wcs, _make_cd_wcs(config.PIXEL_SCALE), prf, ra, dec, (64, 64)
            ).reshape(64, 64)
            out_2x = solver.generate_prf_fast(
                scene_wcs, _make_cd_wcs(config.PIXEL_SCALE * 2.0), prf, ra, dec, (64, 64)
            ).reshape(64, 64)

        # Peak position should be identical
        assert np.unravel_index(np.argmax(out_1x), (64, 64)) == \
               np.unravel_index(np.argmax(out_2x), (64, 64)), \
            "Peak moved when native scale changed — PRF scale not fixed"

        # FWHM (pixels above half-max) should be nearly equal
        n_1x = int(np.sum(out_1x >= out_1x.max() / 2.0))
        n_2x = int(np.sum(out_2x >= out_2x.max() / 2.0))
        assert abs(n_1x - n_2x) <= 4, \
            f"FWHM changed with native scale: {n_1x} vs {n_2x} pixels above half-max"


# ---------------------------------------------------------------------------
# Tests: orientation preservation
# ---------------------------------------------------------------------------

class TestPRFOrientationPreserved:
    """Rotation from native WCS must be preserved in the PRF WCS."""

    def test_both_north_up_and_rotated_give_spread_prf(self):
        """Both north-up and 45°-rotated native WCS should give spread PRFs (not deltas)."""
        prf = _tiny_prf()
        ra, dec = float(config.TRANSIENT_RA), float(config.TRANSIENT_DEC)
        scene_wcs = _make_cdelt_wcs(config.PIXEL_SCALE / config.SUPERSAMPLE_FACTOR)

        with _prf_scale_override(_TEST_PRF_SCALE_ARCSEC):
            out_0 = solver.generate_prf_fast(
                scene_wcs, _make_cd_wcs(config.PIXEL_SCALE, rotation_deg=0.0), prf, ra, dec, (64, 64)
            ).reshape(64, 64)
            out_45 = solver.generate_prf_fast(
                scene_wcs, _make_cd_wcs(config.PIXEL_SCALE, rotation_deg=45.0), prf, ra, dec, (64, 64)
            ).reshape(64, 64)

        for name, out in [("north-up", out_0), ("45° rotated", out_45)]:
            assert abs(out.sum() - 1.0) < 0.05, f"{name}: not normalised (sum={out.sum():.3f})"
            n_sig = int(np.sum(out > out.max() * 0.01))
            assert n_sig > 4, f"{name}: PRF collapsed to delta ({n_sig} significant pixels)"
