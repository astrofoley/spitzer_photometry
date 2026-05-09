"""Pipeline through joint MAP solve (shared by main.py and standalone tools).

Native-fit path (`FIT_ON_NATIVE_PIXELS`): optional square crop (`NATIVE_CUTOUT_SIZE`), local CR mask,
then optional ``unmask_sigma_inf_in_radius`` before star selection and ``solver.run_gls_solve``.
GP hyperparameters are optimized only when ``USE_SCENE_GP_PRIOR`` and ``GP_OPTIMIZE_HYPERPARAMS``
are both enabled.
"""
import os
import warnings

import numpy as np
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy.utils.exceptions import AstropyWarning

from . import config, diagnostics, gp_model, native_fit_campaign, preprocessing, solver

warnings.simplefilter('ignore', category=AstropyWarning)


def run_pipeline_fit_core(*, skip_pre_analysis_check: bool = False):
    """
    Run catalog → alignment → cutouts → CR flag → star detect → joint solve.
    Does not reconstruct full model or write diagnostics PDFs.

    Returns None on failure, else dict with keys:
      cutouts, cutout_wcs, results, all_stars, full_flux_list, median_stamp,
      stretch_mask, chan_str, target, n_epochs
    """
    print("=== Pipeline fit stage ===")
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    os.makedirs(config.DIAGNOSTIC_DIR, exist_ok=True)

    print("\n[Init] Finding Files...")
    all_files = preprocessing.find_spitzer_files(config.DATA_DIR)

    print("\n[Step 0] Initializing Source Catalog...")
    preprocessing.get_or_create_source_catalog(all_files)

    sci_files, tpl_files = preprocessing.categorize_observations(all_files, config.SPLIT_DATE_MJD)
    if not tpl_files:
        print("No template files; abort.")
        return None

    with fits.open(tpl_files[0]['image']) as hdul:
        chan_str = 'ch2' if 'ch2' in tpl_files[0]['image'] else 'ch1'

    full_list = []
    for f in sci_files:
        f['is_template'] = False
        full_list.append(f)
    for f in tpl_files:
        f['is_template'] = True
        full_list.append(f)
    full_list.sort(key=lambda x: x.get('mjd', 0))

    curr_epoch = 0
    last_mjd = full_list[0]['mjd']
    for f in full_list:
        if (f['mjd'] - last_mjd) > config.EPOCH_WINDOW_DAYS:
            curr_epoch += 1
        f['epoch_id'] = curr_epoch
        last_mjd = f['mjd']
    n_epochs = curr_epoch + 1

    target = SkyCoord(config.TRANSIENT_RA, config.TRANSIENT_DEC, unit='deg')

    print("\n[Step 1] Initial Full-Field Reprojection...")
    mosaic_wcs, mosaic_shape = preprocessing.define_mosaic_wcs(full_list, target)
    tpl_list_sub = [f for f in full_list if f['is_template']]
    processed_tpl = preprocessing.reproject_to_grid(tpl_list_sub, mosaic_wcs, mosaic_shape)

    print("\n[Step 2] Creating Median Template Stack...")
    tpl_cube = np.array([p['data'] for p in processed_tpl])
    med_stack, mad_stack = preprocessing.create_median_stack(tpl_cube)

    print("\n[Step 2b] Detecting Sources on Deep Template...")
    template_stars_table = preprocessing.update_catalog_with_template(
        med_stack, mosaic_wcs, 'deep_template',
    )

    print("\n[Step 3] Loading Full Source Catalog for Alignment...")
    source_cat = preprocessing.get_or_create_source_catalog(all_files)

    tpl_stars = []
    tpl_fluxes = []
    if template_stars_table is not None:
        tpl_stars = [SkyCoord(r['ra'], r['dec'], unit='deg') for r in template_stars_table]
        tpl_fluxes = template_stars_table['flux']

    if not skip_pre_analysis_check:
        deep_stretch = np.isfinite(np.asanyarray(med_stack)) & (np.asanyarray(med_stack) != 0)
        diagnostics.plot_pre_analysis_check(
            med_stack, mosaic_wcs, tpl_stars, tpl_fluxes, target, stretch_mask=deep_stretch,
        )

    fits.writeto(
        os.path.join(config.OUTPUT_DIR, f'deep_template_{chan_str}.fits'),
        med_stack, mosaic_wcs.to_header(), overwrite=True,
    )
    fits.writeto(
        os.path.join(config.OUTPUT_DIR, f'deep_variance_{chan_str}.fits'),
        mad_stack**2, mosaic_wcs.to_header(), overwrite=True,
    )

    print("\n[Step 5-6] Astrometric Alignment (KDTree)...")
    preprocessing.align_frames_to_template(full_list, med_stack, mosaic_wcs, source_cat)

    print("\n[Step 9] Extracting Analysis Cutouts...")
    fit_on_native = bool(getattr(config, 'FIT_ON_NATIVE_PIXELS', False))
    if fit_on_native:
        cutouts, cutout_wcs = preprocessing.extract_native_analysis_cutouts(full_list, target)
        native_cutout_size = int(getattr(config, 'NATIVE_CUTOUT_SIZE', 0))
        # Tighter ROI around the transient for cost / conditioning; WCS slices stay consistent.
        if native_cutout_size > 0:
            cutouts = [
                native_fit_campaign.crop_cutout_to_size(
                    c, native_cutout_size, float(config.TRANSIENT_RA), float(config.TRANSIENT_DEC)
                )
                for c in cutouts
            ]
            cutout_wcs = cutouts[0]['wcs']
    else:
        cutouts, cutout_wcs = preprocessing.extract_analysis_cutouts(full_list, target)

    print("\n[Step 8] Final CR Pass on Cutouts...")
    tpl_cutouts = [c['data'] for c in cutouts if c['is_template']]
    if tpl_cutouts:
        median_stamp = np.nanmedian(np.array(tpl_cutouts), axis=0)
    else:
        median_stamp = np.zeros(cutouts[0]['data'].shape)

    if fit_on_native:
        for c in cutouts:
            native_fit_campaign.apply_native_cutout_cr_mask(c)
            # In-place: recover pixels that were already inf-sigma in the FITS (not from our CR pass).
            native_fit_campaign.unmask_sigma_inf_in_radius(
                c,
                float(getattr(config, 'UNMASK_SIGMA_INF_RADIUS_PX', 0.0)),
                str(getattr(config, 'UNMASK_SIGMA_INF_CENTER', 'nuclear')),
            )
        print("   [Native fit] CR pixels set to inf sigma (masked from fit); using template star list.")
        raw_stars_table = template_stars_table
    else:
        preprocessing.flag_cosmic_rays(cutouts, median_stamp, cutout_wcs, target)
        print("   Detecting stars in analysis cutout (Robust Method)...")
        raw_stars_table = preprocessing.detect_sources(
            median_stamp, cutout_wcs, sigma_map=None, is_template=True, mask_nucleus=False,
        )

    solver_stars = []
    solver_init_fluxes = []

    print(f"\n   [Star Detection Report]")
    print(f"   {'Index':<5} {'Dist(as)':<10} {'Init Flux':<12} {'Category'}")
    print(f"   {'-'*40}")

    if raw_stars_table is not None:
        for i, row in enumerate(raw_stars_table):
            sc = SkyCoord(row['ra'], row['dec'], unit='deg')
            flux_val = float(row['flux'])
            dist = sc.separation(target).arcsec
            if fit_on_native and dist > 40.0:
                # Keep only stars near the analysis window for native stamps.
                continue
            print(f"   {i:<5} {dist:<10.3f} {flux_val:<12.4e} Solver (all in joint fit)")
            solver_stars.append(sc)
            solver_init_fluxes.append(flux_val)

    print(f"   Total Detections: {len(raw_stars_table) if raw_stars_table else 0}")
    print(f"   Solver Stars:     {len(solver_stars)}")

    print(f"\n[Step 10] Solving Joint System ({len(cutouts)} frames)...")
    vals = [c['data'][c['data'] != 0] for c in cutouts]
    if vals:
        flat = np.concatenate(vals)
        low, high = np.percentile(flat, [16, 84])
        var_est = ((high - low) / 2.0) ** 2
    else:
        var_est = 1e-7

    tpl_cutout_entries = [c for c in cutouts if c.get('is_template')]
    gp_ell, gp_var = config.INIT_LENGTH_SCALE, max(var_est, config.INIT_VARIANCE)
    if (
        bool(getattr(config, 'USE_SCENE_GP_PRIOR', True))
        and getattr(config, 'GP_OPTIMIZE_HYPERPARAMS', False)
        and tpl_cutout_entries
    ):
        gp_ell, gp_var = gp_model.optimize_hyperparameters(tpl_cutout_entries)
    gp_params = {'ell': gp_ell, 'var': gp_var}

    results = solver.run_gls_solve(
        cutouts,
        solver_stars,
        solver_init_fluxes,
        gp_params,
        (gp_ell, gp_var),
        median_stamp,
        cutout_wcs,
        n_epochs,
    )

    if results is None:
        print("CRITICAL ERROR: Solver returned None.")
        return None

    fitted_fluxes = results['star_fluxes']
    final_star_fluxes = []
    for i, flux in enumerate(fitted_fluxes):
        if flux <= 0.0:
            print(
                f"   [Warning] Solver returned {flux:.2e} for Star {i}. "
                f"Using Init Flux: {solver_init_fluxes[i]:.2e}",
            )
            final_star_fluxes.append(solver_init_fluxes[i])
        else:
            final_star_fluxes.append(flux)

    full_flux_list = np.array(final_star_fluxes)
    results['star_fluxes'] = full_flux_list

    print("\n   [Final Fluxes for Reconstruction]")
    for i, flux in enumerate(full_flux_list):
        print(f"   Star {i}: {flux:.4e} Jy")

    stretch_mask = diagnostics.diagnostic_stretch_mask(cutouts, cutouts[0]['data'].shape)

    return {
        'cutouts': cutouts,
        'cutout_wcs': cutout_wcs,
        'results': results,
        'all_stars': solver_stars,
        'full_flux_list': full_flux_list,
        'median_stamp': median_stamp,
        'stretch_mask': stretch_mask,
        'chan_str': chan_str,
        'target': target,
        'n_epochs': n_epochs,
    }
