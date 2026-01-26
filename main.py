"""main.py"""
import os
import sys
import warnings
import numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy.coordinates import SkyCoord
from src import config, preprocessing, solver, diagnostics, plotting

# Suppress annoying FITS warnings
warnings.filterwarnings('ignore', category=UserWarning, append=True)
from astropy.utils.exceptions import AstropyWarning
warnings.simplefilter('ignore', category=AstropyWarning)

def run_pipeline():
    print("=== Spitzer Photometry Pipeline ===")
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    os.makedirs(config.DIAGNOSTIC_DIR, exist_ok=True)

    # --- Setup ---
    print("\n[Init] Finding Files...")
    all_files = preprocessing.find_spitzer_files(config.DATA_DIR)
    
    print("\n[Step 0] Initializing Source Catalog...")
    preprocessing.get_or_create_source_catalog(all_files)
    
    sci_files, tpl_files = preprocessing.categorize_observations(all_files, config.SPLIT_DATE_MJD)
    if not tpl_files: return
    
    with fits.open(tpl_files[0]['image']) as hdul:
        chan_str = 'ch2' if 'ch2' in tpl_files[0]['image'] else 'ch1'

    full_list = []
    for f in sci_files: f['is_template'] = False; full_list.append(f)
    for f in tpl_files: f['is_template'] = True; full_list.append(f)
    full_list.sort(key=lambda x: x.get('mjd', 0))
    
    curr_epoch = 0; last_mjd = full_list[0]['mjd']
    for f in full_list:
        if (f['mjd'] - last_mjd) > config.EPOCH_WINDOW_DAYS: curr_epoch += 1
        f['epoch_id'] = curr_epoch
        last_mjd = f['mjd']
    n_epochs = curr_epoch + 1
    
    target = SkyCoord(config.TRANSIENT_RA, config.TRANSIENT_DEC, unit='deg')

    # --- Step 1: Reprojection ---
    print("\n[Step 1] Initial Full-Field Reprojection...")
    mosaic_wcs, mosaic_shape = preprocessing.define_mosaic_wcs(full_list, target)
    tpl_list_sub = [f for f in full_list if f['is_template']]
    processed_tpl = preprocessing.reproject_to_grid(tpl_list_sub, mosaic_wcs, mosaic_shape)
    
    # --- Step 2: Stack ---
    print("\n[Step 2] Creating Median Template Stack...")
    tpl_cube = np.array([p['data'] for p in processed_tpl])
    med_stack, mad_stack = preprocessing.create_median_stack(tpl_cube)
    
    # --- Step 2b: Update Catalog with Deep Template ---
    print("\n[Step 2b] Detecting Sources on Deep Template...")
    template_stars_table = preprocessing.update_catalog_with_template(med_stack, mosaic_wcs, 'deep_template')
    
    # --- Step 3: RELOAD CATALOG ---
    print("\n[Step 3] Loading Full Source Catalog for Alignment...")
    source_cat = preprocessing.get_or_create_source_catalog(all_files)
    
    # --- Diagnostics: Pre-Analysis ---
    # FIXED: Extract stars/fluxes from table to pass to new function signature
    tpl_stars = []
    tpl_fluxes = []
    if template_stars_table is not None:
        tpl_stars = [SkyCoord(r['ra'], r['dec'], unit='deg') for r in template_stars_table]
        tpl_fluxes = template_stars_table['flux']

    diagnostics.plot_pre_analysis_check(med_stack, mosaic_wcs, tpl_stars, tpl_fluxes, target)
    
    fits.writeto(os.path.join(config.OUTPUT_DIR, f'deep_template_{chan_str}.fits'), med_stack, mosaic_wcs.to_header(), overwrite=True)
    fits.writeto(os.path.join(config.OUTPUT_DIR, f'deep_variance_{chan_str}.fits'), mad_stack**2, mosaic_wcs.to_header(), overwrite=True)
    
    # --- Step 5-6: Alignment ---
    print("\n[Step 5-6] Astrometric Alignment (KDTree)...")
    preprocessing.align_frames_to_template(full_list, med_stack, mosaic_wcs, source_cat)
    
    # --- Step 9: Cutouts ---
    print("\n[Step 9] Extracting Analysis Cutouts...")
    cutouts, cutout_wcs = preprocessing.extract_analysis_cutouts(full_list, target)
    
    # --- Step 8: Final CR Pass & Solver Stars ---
    print("\n[Step 8] Final CR Pass on Cutouts...")
    tpl_cutouts = [c['data'] for c in cutouts if c['is_template']]
    if tpl_cutouts:
        final_template = np.nanmedian(np.array(tpl_cutouts), axis=0)
    else: final_template = np.zeros(cutouts[0]['data'].shape)
    
    preprocessing.flag_cosmic_rays(cutouts, final_template, cutout_wcs, target)
    
    print("   Detecting stars in analysis cutout (Robust Method)...")
    raw_stars_table = preprocessing.detect_sources(final_template, cutout_wcs, sigma_map=None, is_template=True, mask_nucleus=False)
    
    solver_stars = []
    solver_init_fluxes = []
    extra_stars = []
    extra_fluxes = []
    
    print(f"\n   [Star Detection Report]")
    print(f"   {'Index':<5} {'Dist(as)':<10} {'Init Flux':<12} {'Category'}")
    print(f"   {'-'*40}")
    
    if raw_stars_table is not None:
        for i, row in enumerate(raw_stars_table):
            sc = SkyCoord(row['ra'], row['dec'], unit='deg')
            flux_val = float(row['flux'])
            dist = sc.separation(target).arcsec
            
            cat = "Solver (Yellow)" if dist > 1.5 else "Fixed (Cyan)"
            print(f"   {i:<5} {dist:<10.3f} {flux_val:<12.4e} {cat}")
            
            if dist > 1.5:
                solver_stars.append(sc)
                solver_init_fluxes.append(flux_val)
            else:
                extra_stars.append(sc)
                extra_fluxes.append(flux_val)
                
    print(f"   Total Detections: {len(raw_stars_table) if raw_stars_table else 0}")
    print(f"   Solver Stars:     {len(solver_stars)}")
    print(f"   Fixed Stars:      {len(extra_stars)}")

    # --- Step 10: Solve ---
    print(f"\n[Step 10] Solving Joint System ({len(cutouts)} frames)...")
    vals = [c['data'][c['data']!=0] for c in cutouts]
    if vals:
        flat = np.concatenate(vals)
        low, high = np.percentile(flat, [16, 84])
        var_est = ((high-low)/2.0)**2
    else: var_est = 1e-7
    
    results = solver.run_gls_solve(
        cutouts,
        solver_stars,
        solver_init_fluxes,
        {},
        (8.0, var_est),
        final_template,
        cutout_wcs,
        n_epochs
    )
    
    if results is None:
        print("CRITICAL ERROR: Solver returned None.")
        return

    fitted_fluxes = results['star_fluxes']
    final_solver_fluxes = []
    
    for i, flux in enumerate(fitted_fluxes):
        if flux <= 0.0:
            print(f"   [Warning] Solver returned {flux:.2e} for Star {i}. Using Init Flux: {solver_init_fluxes[i]:.2e}")
            final_solver_fluxes.append(solver_init_fluxes[i])
        else:
            final_solver_fluxes.append(flux)
            
    full_flux_list = np.concatenate([np.array(final_solver_fluxes), np.array(extra_fluxes)])
    all_stars = solver_stars + extra_stars
    
    print("\n   [Final Fluxes for Reconstruction]")
    for i, flux in enumerate(full_flux_list):
        print(f"   Star {i}: {flux:.4e} Jy")

    results['star_fluxes'] = full_flux_list
    
    print("   Reconstructing Full Model...")
    full_template = solver.reconstruct_full_model(results, all_stars, full_flux_list, chan_str, cutouts)
    results['model_scene'] = full_template
    
    epoch_bgs = results['epoch_backgrounds']
    frame_bgs = np.array([epoch_bgs[c['epoch_id']] for c in cutouts])
    results['backgrounds'] = frame_bgs
    
    # --- Final Diagnostics ---
    diagnostics.plot_fit_template_with_stars(full_template, results['scene_wcs'], all_stars, full_flux_list, target)
    
    # NOTE: Pass med_stack as the "Deep Stack" reference for residual plots
    diagnostics.plot_bcd_residuals(cutouts, results, all_stars, full_flux_list, final_template)
    
    diagnostics.plot_epoch_stacks(cutouts, results, target)
    
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
    
    diagnostics.plot_lightcurve(t, chan_str)
    
    print("Done.")

if __name__ == "__main__":
    run_pipeline()
