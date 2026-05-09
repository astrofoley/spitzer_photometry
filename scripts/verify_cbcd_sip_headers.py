#!/usr/bin/env python3
"""
Quickly inspect WCS SIP / distortion on FITS under ``src.config.DATA_DIR`` (raw CBCDs).

Run after placing real Spitzer/IRAC products in ``data/raw`` to confirm ``WCS.sip``
loads when headers use ``*-TAN-SIP`` CTYPEs. This does not modify any data.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from astropy.io import fits  # noqa: E402
from astropy.wcs import WCS  # noqa: E402

from src import config  # noqa: E402


def main() -> int:
    root = Path(config.DATA_DIR)
    paths = sorted(root.rglob("*.fits"))
    if not paths:
        print(f"No FITS under {root} — add raw CBCDs here to verify SIP headers locally.")
        return 0
    max_files = 20
    for path in paths[:max_files]:
        try:
            with fits.open(path, memmap=False) as hdul:
                hdr = hdul[0].header
        except OSError as e:
            print(f"{path}: open failed: {e}")
            continue
        w = WCS(hdr)
        c1 = str(hdr.get("CTYPE1", ""))
        c2 = str(hdr.get("CTYPE2", ""))
        sip_loaded = w.sip is not None
        has_dist = getattr(w, "has_distortion", False)
        print(f"{path.name}: CTYPE=({c1!r},{c2!r}) sip={sip_loaded} has_distortion={has_dist}")
    if len(paths) > max_files:
        print(f"(Listed first {max_files} of {len(paths)} files.)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
