"""
Native-detector forward-model scaffolding (plan: model warped to data, not data to model).

Full multi-epoch native likelihood requires a per-frame forward operator F_i coupling a fiducial
sky scene to each BCD pixel grid. Until that exists, use extract_native_stamp_for_target only
for experiments/tests; the main pipeline keeps reprojected analysis stamps when USE_NATIVE_* is off.
"""
from __future__ import annotations

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.nddata import Cutout2D
from astropy.wcs import WCS


def extract_native_stamp_for_target(
    data: np.ndarray,
    uncertainty: np.ndarray,
    wcs: WCS,
    target: SkyCoord,
    size_xy,
    *,
    fill_value: float = np.nan,
    fill_sigma: float = np.inf,
):
    """
    Centered rectangular cutout in **native** BCD pixels (no interpolation of values).

    Returns
    -------
    data_cut, sigma_cut, wcs_cut : arrays and WCS of the stamp
    """
    cut_d = Cutout2D(
        data,
        position=target,
        size=size_xy,
        wcs=wcs,
        mode="partial",
        fill_value=fill_value,
    )
    cut_s = Cutout2D(
        uncertainty,
        position=target,
        size=size_xy,
        wcs=wcs,
        mode="partial",
        fill_value=fill_sigma,
    )
    return cut_d.data, cut_s.data, cut_d.wcs
