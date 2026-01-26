"""
diagnostic_tool.py
Run this to benchmark the solver and visualize the system matrix.
"""
import time
import cProfile
import pstats
import io
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from src import config, preprocessing, solver, gp_model
from astropy.coordinates import SkyCoord

def benchmark_solver():
    print("=== PIPELINE BENCHMARK & DIAGNOSTICS ===")
    
    # 1. Load Data (Subset)
    print("Loading subset of data...")
    all_files = preprocessing.find_spitzer_files(config.DATA_DIR)
    science_files, template_files = preprocessing.categorize_observations(all_files, config.SPLIT_DATE_MJD)
    
    # Take only 10 science frames for speed
    science_subset = science_files[:10]
    target = SkyCoord(config.TRANSIENT_RA, config.TRANSIENT_DEC, unit='deg')
    
    loaded_sci = []
    for f in science_subset:
        try: loaded_sci.append(preprocessing.load_data(f, target, config.STAMP_SIZE_ARCSEC))
        except: pass
        
    print(f"Benchmarking with {len(loaded_sci)} frames...")
    
    # Generate Fake Template (Fast)
    # We just need the WCS and a dummy array
    temp_ref = preprocessing.load_data(template_files[0], size_arcsec=None) # Full frame reference
    w_native = temp_ref['wcs']
    # Create Native Grid
    w_native.wcs.crval = [config.TRANSIENT_RA, config.TRANSIENT_DEC]
    h, w = temp_ref['data'].shape
    w_native.wcs.crpix = [w/2, h/2]
    
    # Super-Res Grid
    w_sr = w_native.deepcopy()
    factor = config.SUPERSAMPLE_FACTOR
    if hasattr(w_sr.wcs, 'cd'): w_sr.wcs.cd /= factor
    else: w_sr.wcs.cdelt /= factor
    w_sr.wcs.crpix = [w/2 * factor, h/2 * factor]
    sr_h, sr_w = int(h * factor), int(w * factor)
    
    deep_temp = np.zeros((sr_h, sr_w)) # Dummy template
    
    # Dummy Stars
    stars = []
    
    # Hyperparams
    ell, var = 1.0, 1.0
    
    # --- START PROFILING ---
    pr = cProfile.Profile()
    pr.enable()
    
    t0 = time.time()
    results = solver.run_gls_solve(loaded_sci, stars, (ell, var), deep_temp, w_sr)
    t1 = time.time()
    
    pr.disable()
    # -----------------------
    
    print(f"\nTotal Solver Time: {t1 - t0:.2f} seconds")
    print(f"Time per BCD: {(t1-t0)/len(loaded_sci):.2f} seconds")
    
    # Print Stats
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumtime')
    ps.print_stats(20) # Top 20 time-consumers
    print("\n--- TOP BOTTLENECKS ---")
    print(s.getvalue())
    
    # --- VISUALIZATION ---
    print("Generating Diagnostic Plot (benchmark_results.pdf)...")
    with PdfPages('benchmark_results.pdf') as pdf:
        
        # 1. Coverage Map
        # Reconstruct coverage from sparse matrices inside solver?
        # Easier: Just plot the model_scene validity (where variance is not infinite)
        # But we don't have variance here.
        # Let's plot the scene model itself (it will be zero, but the structure exists)
        
        # Better: Plot the Footprints manually
        coverage = np.zeros((sr_h, sr_w))
        scene_wcs = results['scene_wcs'] # This is the local scene WCS used in solver
        
        # Project center of each BCD onto Scene
        fig, ax = plt.subplots(figsize=(8,8))
        ax.imshow(coverage, origin='lower', cmap='gray_r')
        
        for d in loaded_sci:
            # Simple check: Convert BCD center to Scene Pixels
            cx, cy = d['header']['NAXIS1']/2, d['header']['NAXIS2']/2
            ra, dec = d['wcs'].pixel_to_world_values(cx, cy)
            sx, sy = scene_wcs.world_to_pixel_values(ra, dec)
            ax.plot(sx, sy, 'r+', markersize=10, alpha=0.5)
            
        ax.set_title(f"Centroids of {len(loaded_sci)} BCDs on Scene Grid")
        pdf.savefig(fig); plt.close()
        
    print("Done.")

if __name__ == "__main__":
    benchmark_solver()
