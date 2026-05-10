"""SIP / projection cache: native WCS uses full Astropy transform; cache keys distinguish SIP."""
import numpy as np
import pytest
from astropy.io import fits
from astropy.wcs import WCS

from src.solver import _sip_distortion_fingerprint, _wcs_key, _wcs_projection_cache_key


def _sip_header(a2_0: float = 1.0e-3, b0_2: float = 1.0e-3) -> fits.Header:
    hdr_txt = f"""SIMPLE  =                    T
BITPIX  =                  -32
NAXIS   =                    2
NAXIS1  =                  512
NAXIS2  =                  512
WCSAXES =                    2
CRPIX1  =                256.5
CRPIX2  =                256.5
PC1_1   =           9.0E-6
PC1_2   =                  0
PC2_1   =                  0
PC2_2   =           8.99E-6
CDELT1  =                  1
CDELT2  =                  1
CRVAL1  =                  50
CRVAL2  =                  50
CTYPE1  = 'RA---TAN-SIP'
CTYPE2  = 'DEC--TAN-SIP'
A_ORDER =                    2
B_ORDER =                    2
A_2_0   =          {a2_0:.8E}
B_0_2   =          {b0_2:.8E}
AP_ORDER=                    0
BP_ORDER=                    0
"""
    return fits.Header.fromstring(hdr_txt, sep="\n")


def _linear_header_from_sip(hdr_sip: fits.Header) -> fits.Header:
    hdr_lin = hdr_sip.copy()
    hdr_lin["CTYPE1"] = "RA---TAN"
    hdr_lin["CTYPE2"] = "DEC--TAN"
    for k in list(hdr_lin.keys()):
        if k.startswith("A_") or k.startswith("B_"):
            hdr_lin.remove(k, ignore_missing=True)
        if k in ("A_ORDER", "B_ORDER", "AP_ORDER", "BP_ORDER"):
            hdr_lin.remove(k, ignore_missing=True)
    return hdr_lin


def test_sip_wcs_loads_distortion_and_changes_pixel_to_sky():
    hdr_sip = _sip_header()
    w_sip = WCS(hdr_sip)
    assert w_sip.sip is not None
    w_lin = WCS(_linear_header_from_sip(hdr_sip))
    assert w_lin.sip is None
    px, py = 100.0, 400.0
    s0, s1 = w_sip.pixel_to_world_values(px, py)
    l0, l1 = w_lin.pixel_to_world_values(px, py)
    assert abs(float(s0 - l0)) > 1e-6 or abs(float(s1 - l1)) > 1e-6


def test_sip_all_pix2world_agrees_with_pixel_to_world():
    w = WCS(_sip_header())
    for px, py in [(0.0, 0.0), (10.5, 300.0)]:
        ra0, dec0 = w.pixel_to_world_values(px, py)
        ra1, dec1 = w.all_pix2world(px, py, 0)
        assert np.allclose([ra0, dec0], [ra1, dec1], rtol=0, atol=1e-9)


def test_projection_cache_key_differs_for_same_linear_different_sip():
    w_a = WCS(_sip_header(a2_0=1e-3, b0_2=1e-3))
    w_b = WCS(_sip_header(a2_0=2e-3, b0_2=2e-3))
    assert _wcs_key(w_a) == _wcs_key(w_b)
    assert _sip_distortion_fingerprint(w_a) != _sip_distortion_fingerprint(w_b)
    assert _wcs_projection_cache_key(w_a) != _wcs_projection_cache_key(w_b)


def test_projection_cache_key_differs_sip_vs_linearized_header():
    """SIP vs stripped linear header: projection cache key must not collide."""
    hdr_sip = _sip_header()
    w_sip = WCS(hdr_sip)
    w_lin = WCS(_linear_header_from_sip(hdr_sip))
    assert _sip_distortion_fingerprint(w_sip) != _sip_distortion_fingerprint(w_lin)
    assert _wcs_projection_cache_key(w_sip) != _wcs_projection_cache_key(w_lin)


def test_real_raw_fits_sip_matches_ctype_when_data_present():
    """
    When raw data exist under ``config.DATA_DIR``, require SIP objects to load for
    FITS headers that declare ``-SIP`` tangent-plane types (e.g. IRAC BCDs).
    Skips if no FITS found or header is not SIP-declared.
    """
    import pathlib

    from src import config

    paths = sorted(pathlib.Path(config.DATA_DIR).rglob("*.fits"))
    if not paths:
        pytest.skip("No FITS under config.DATA_DIR")
    path = paths[0]
    with fits.open(path, memmap=False) as hdul:
        hdr = hdul[0].header
    c1 = str(hdr.get("CTYPE1", ""))
    c2 = str(hdr.get("CTYPE2", ""))
    if "SIP" not in (c1 + c2):
        pytest.skip(f"{path}: CTYPE not SIP-style; skipping SIP load check")
    w = WCS(hdr)
    assert w.sip is not None, "CTYPE declares SIP but WCS.sip is None"

