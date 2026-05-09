"""
diagnostic_tool.py
Benchmark the solver on synthetic cutouts (no data required) or on a small real subset.
"""
import time
import io
import cProfile
import pstats
import numpy as np
from astropy.wcs import WCS

from src import config, solver


def make_synthetic_cutouts(n_frames=4, n_pix=24, seed=0):
    """Minimal cutout dicts matching preprocessing/solver expectations."""
    rng = np.random.default_rng(seed)
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
            'is_template': (i >= n_frames // 2),
        })
    return cutouts, w_small


def benchmark_solver(n_frames=4, n_pix=24, profile=False):
    print("=== Solver benchmark (synthetic cutouts) ===")
    cutouts, cutout_wcs = make_synthetic_cutouts(n_frames=n_frames, n_pix=n_pix)
    stars = []
    star_fluxes = []
    ell, var = 4.0, 1e-8
    n_epochs = 1

    if profile:
        pr = cProfile.Profile()
        pr.enable()

    t0 = time.time()
    results = solver.run_gls_solve(
        cutouts,
        stars,
        star_fluxes,
        {'ell': ell, 'var': var},
        (ell, var),
        np.zeros((n_pix, n_pix)),
        cutout_wcs,
        n_epochs,
    )
    t1 = time.time()

    if profile:
        pr.disable()
        s = io.StringIO()
        pstats.Stats(pr, stream=s).sort_stats('cumtime').print_stats(15)
        print("\n--- Profile (top 15) ---\n", s.getvalue())

    print(f"Solve time: {t1 - t0:.3f} s ({len(cutouts)} frames)")
    if results is None:
        print("Solver returned None")
        return None
    print(f"transient flux (first): {results['transient_fluxes'][0]:.3e}")
    print(f"transient err (first):   {results['transient_errs'][0]:.3e}")
    return results


if __name__ == "__main__":
    benchmark_solver(profile=False)
