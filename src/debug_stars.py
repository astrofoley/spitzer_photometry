"""src/debug_stars.py"""
import os
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from astropy.visualization import ZScaleInterval
from src import config, preprocessing

def run_star_check():
    print("Running Star Position Diagnostic...")
    
    # 1. Load Data
    chan_str = 'ch1' # Default to ch1
    template_path = os.path.join(config.OUTPUT_DIR, f'deep_template_{chan_str}.fits')
    if not os.path.exists(template_path):
        # Fallback to try ch2 if ch1 missing
        chan_str = 'ch2'
        template_path = os.path.join(config.OUTPUT_DIR, f'deep_template_{chan_str}.fits')
        if not os.path.exists(template_path):
            print(f"No template found. Run main pipeline first.")
            return

    print(f"Loading {template_path}...")
    with fits.open(template_path) as hdul:
        data = hdul[0].data
        header = hdul[0].header
        wcs = WCS(header)
        
    # 2. Re-run Detection (Same settings as main pipeline)
    print("Running Source Extractor...")
    raw_star_coords = preprocessing.detect_stars(data, wcs, mask_radius_arcsec=15.0)
    
    target_coord = SkyCoord(config.TRANSIENT_RA, config.TRANSIENT_DEC, unit='deg')
    
    # 3. Plotting
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Use ZScale for better contrast of faint structure
    interval = ZScaleInterval()
    vmin, vmax = interval.get_limits(data)
    ax.imshow(data, origin='lower', cmap='gray', vmin=vmin, vmax=vmax)
    
    print(f"Transient Location: {target_coord.to_string('hmsdms')}")
    
    # Plot Transient Location
    tx, ty = wcs.world_to_pixel(target_coord)
    ax.scatter(tx, ty, marker='+', s=300, c='cyan', label='Transient Loc', linewidth=2)
    
    # Check separations
    culprit_found = False
    
    print("\n--- Separation Check ---")
    for i, s in enumerate(raw_star_coords):
        sx, sy = wcs.world_to_pixel(s)
        sep = s.separation(target_coord).arcsec
        
        if sep < 2.5: # Using 2.5 arcsec as the critical threshold
            print(f"!!! CULPRIT FOUND !!! Star #{i} at distance {sep:.3f} arcsec")
            ax.scatter(sx, sy, s=200, facecolors='none', edgecolors='red', linewidth=3, label=f'CULPRIT (<2.5")')
            culprit_found = True
        else:
            ax.scatter(sx, sy, s=50, facecolors='none', edgecolors='lime', alpha=0.5)

    if not culprit_found:
        print("PASS: No stars detected within 2.5 arcsec of transient.")
        ax.text(0.05, 0.95, "PASS: No Central Star", transform=ax.transAxes, color='lime', fontsize=16, fontweight='bold')
    else:
        ax.text(0.05, 0.95, "FAIL: Central Star Detected", transform=ax.transAxes, color='red', fontsize=16, fontweight='bold')
        
    # Legend handling
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='upper right')
    
    out_path = os.path.join(config.DIAGNOSTIC_DIR, 'DEBUG_STAR_POSITIONS.png')
    ax.set_title(f"Star Detection Diagnostic\n{os.path.basename(template_path)}")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"\nSaved diagnostic plot to {out_path}")

if __name__ == "__main__":
    run_star_check()
