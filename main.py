"""Top-level pipeline: joint fit + standard diagnostics products.

`run_pipeline()` applies a **nominal science configuration** via
`native_fit_campaign._temporary_config` (see `nominal_overrides` below and
`docs/NOMINAL_NATIVE_SCIENCE_RUN.md`) so one command reproduces the intended
native SR=2, PRF-on, GP-off, transient-float production run without editing
every default in `src/config.py` for ad-hoc tests.
"""
import os
import warnings
import numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy.coordinates import SkyCoord
from src import config, solver, diagnostics, pipeline_fit
from src.native_fit_campaign import _temporary_config

# Suppress annoying FITS warnings
warnings.filterwarnings('ignore', category=UserWarning, append=True)
from astropy.utils.exceptions import AstropyWarning
warnings.simplefilter('ignore', category=AstropyWarning)

def run_pipeline():
    print("=== Spitzer Photometry Pipeline ===")
    # Nominal science settings (see docs/NOMINAL_NATIVE_SCIENCE_RUN.md). The
    # whole fit and diagnostics execute inside this `with` so `config` is
    # patched for the run and restored when `run_pipeline()` returns.
    nominal_overrides = {
        'FIT_ON_NATIVE_PIXELS': True,
        'SUPERSAMPLE_FACTOR': 2,
        'NATIVE_CUTOUT_SIZE': 40,
        'USE_SCENE_GP_PRIOR': False,
        'SCENE_INDEPENDENT_RIDGE': 1e-12,
        'FLOAT_TRANSIENT_POSITION': True,
        'PRF_ORDER_PROJECT_THEN_CONVOLVE': True,
        'PRF_GLS_LTWL_FULL_MAX_PIXELS': 20000,
        'CR_BRIGHT_CORE_GUARD_PERCENTILE': 97.0,
        'CR_BRIGHT_CORE_GUARD_DILATION': 3,
        'CR_BRIGHT_CORE_GUARD_RADIUS_PX': 8.0,
        'CR_BRIGHT_CORE_GUARD_CENTER': 'nuclear',
        'UNMASK_SIGMA_INF_RADIUS_PX': 8.0,
        'UNMASK_SIGMA_INF_CENTER': 'nuclear',
    }
    with _temporary_config(nominal_overrides):
        fit = pipeline_fit.run_pipeline_fit_core()
        if fit is None:
            return

        cutouts = fit['cutouts']
        cutout_wcs = fit['cutout_wcs']
        results = fit['results']
        all_stars = fit['all_stars']
        full_flux_list = fit['full_flux_list']
        final_template = fit['median_stamp']
        stretch_mask = fit['stretch_mask']
        chan_str = fit['chan_str']
        target = fit['target']

        print("   Reconstructing Full Model...")
        full_template = solver.reconstruct_full_model(results, all_stars, full_flux_list, chan_str, cutouts)
        results['model_scene'] = full_template
        
        epoch_bgs = results['epoch_backgrounds']
        frame_bgs = np.array([epoch_bgs[c['epoch_id']] for c in cutouts])
        results['backgrounds'] = frame_bgs
        
        # --- Final Diagnostics ---
        stretch_mask = diagnostics.diagnostic_stretch_mask(cutouts, final_template.shape)
        overlay_frame = None
        te_arr = results.get('transient_epoch_fluxes')
        se_arr = results.get('science_epoch_ids')
        if te_arr is not None and len(np.asarray(te_arr)) > 0 and se_arr is not None:
            imax = int(np.argmax(np.asarray(te_arr, dtype=float)))
            eid_pick = int(np.asarray(se_arr)[imax])
            for ii, c in enumerate(cutouts):
                if not c.get('is_template') and int(c['epoch_id']) == eid_pick:
                    overlay_frame = ii
                    break
        if overlay_frame is None:
            for ii, c in enumerate(cutouts):
                if not c.get('is_template'):
                    overlay_frame = ii
                    break

        diagnostics.plot_fit_template_with_stars(
            full_template,
            results['scene_wcs'],
            all_stars,
            full_flux_list,
            target,
            stretch_mask=stretch_mask,
            results=results,
            cutouts=cutouts,
            transient_overlay_frame_index=overlay_frame,
        )

        # NOTE: final_template = median template BCD stamp for "deep stack" reference in residuals
        diagnostics.plot_bcd_residuals(
            cutouts, results, all_stars, full_flux_list, final_template,
            stretch_mask=stretch_mask,
        )

        diagnostics.plot_epoch_stacks(
            cutouts, results, target, all_stars, full_flux_list, stretch_mask=stretch_mask
        )
        # Complements epoch stacks: fixed scene grid median of (data−pred)
        # with transient OFF vs ON in the predictor.
        diagnostics.plot_stacked_residuals_with_without_transient(
            cutouts, results, all_stars, full_flux_list, stretch_mask=stretch_mask
        )
        diagnostics.plot_template_component_stacks(
            cutouts, results, all_stars, full_flux_list, stretch_mask=stretch_mask
        )
        diagnostics.plot_template_prf_vs_residual_orientation(
            cutouts, results, all_stars, full_flux_list, stretch_mask=stretch_mask
        )
        diagnostics.plot_gp_vs_stars(cutouts, results, all_stars, full_flux_list, stretch_mask=stretch_mask)
        
        t = Table([
            [c['mjd'] for c in cutouts],
            results['transient_fluxes'],
            results.get('transient_errs', np.zeros(len(cutouts))),
            frame_bgs,
            [c['epoch_id'] for c in cutouts],
            [c['filename'] for c in cutouts],
            [1 if c.get('is_template') else 0 for c in cutouts]
        ], names=('MJD', 'Flux_Jy', 'Flux_Err_Jy', 'Background_Jy', 'Epoch_ID', 'Filename', 'Is_Template'))
        
        out_csv = os.path.join(config.OUTPUT_DIR, f'lightcurve_{chan_str}.csv')
        t.write(out_csv, overwrite=True)
        fits.writeto(os.path.join(config.OUTPUT_DIR, f'template_model_scene_{chan_str}.fits'), full_template, results['scene_wcs'].to_header(), overwrite=True)
        fits.writeto(
            os.path.join(config.OUTPUT_DIR, f'gp_scene_only_{chan_str}.fits'),
            np.asarray(results['gp_scene']),
            results['scene_wcs'].to_header(),
            overwrite=True,
        )
        
        diagnostics.plot_lightcurve(t, chan_str)

        diagnostics.write_fit_quality_report(
            cutouts, results, all_stars, full_flux_list, final_template, chan_str
        )
        
        print("Done.")

if __name__ == "__main__":
    run_pipeline()
