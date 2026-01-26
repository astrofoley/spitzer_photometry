"""src/plotting.py"""
import os
import numpy as np
import matplotlib.pyplot as plt
from astropy.visualization import ImageNormalize, AsinhStretch, ZScaleInterval
from . import config

def plot_input_diagnostics(sample_bcd, deep_template, deep_var, tpl_wcs, star_coords=None):
    """
    Pre-Analysis Diagnostic Check.
    Overlays detected stars on the Deep Template.
    """
    print("Generating Pre-Analysis Diagnostic PDF (with Star Overlay)...")
    out_pdf = os.path.join(config.DIAGNOSTIC_DIR, 'PRE_ANALYSIS_CHECK.pdf')
    
    from matplotlib.backends.backend_pdf import PdfPages
    
    with PdfPages(out_pdf) as pdf:
        # Page 1: Deep Template & Variance
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # 1. Deep Template
        norm_tpl = ImageNormalize(deep_template, interval=ZScaleInterval(), stretch=AsinhStretch())
        axes[0].imshow(deep_template, origin='lower', cmap='gray', norm=norm_tpl)
        axes[0].set_title("Deep Template (North Up)\n[ZScale + Asinh]")
        
        # Overlay Stars
        if star_coords:
            try:
                # Convert SkyCoords to Pixel Coords
                ra = [s.ra.deg for s in star_coords]
                dec = [s.dec.deg for s in star_coords]
                x, y = tpl_wcs.world_to_pixel_values(np.array(ra), np.array(dec))
                
                # Plot
                axes[0].scatter(x, y, edgecolor='red', facecolor='none', s=80, linewidth=1.5, label=f'Detected ({len(star_coords)})')
                axes[0].legend(loc='upper right')
            except Exception as e:
                print(f"Warning: Failed to plot stars on diagnostic: {e}")
        
        # 2. Variance
        interval_var = ZScaleInterval()
        v_vmin, v_vmax = interval_var.get_limits(deep_var)
        axes[1].imshow(deep_var, origin='lower', cmap='viridis', vmin=v_vmin, vmax=v_vmax)
        axes[1].set_title("Template Variance")
        
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close()
        
    print(f"Saved {out_pdf}")
