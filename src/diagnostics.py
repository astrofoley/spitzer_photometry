"""src/diagnostics.py"""
import os
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from . import config

def get_robust_limits(data, pmin=1, pmax=99):
    valid = data[np.isfinite(data) & (data != 0)]
    if len(valid) == 0: return 0, 1
    lo, hi = np.percentile(valid, [pmin, pmax])
    if hi == lo: hi += 1e-6
    return lo, hi

def plot_pre_analysis_check(stack, wcs, stars, fluxes, target_coord):
    """
    Plots ONLY the deep template with detected stars marked.
    """
    plt.figure(figsize=(12, 12))
    vmin, vmax = get_robust_limits(stack, 0.5, 99.5)
    
    # Log scale for better dynamic range visualization
    if vmax > 0 and vmin > 0:
        norm_data = np.log10(np.clip(stack, vmin, vmax))
    else:
        norm_data = stack
        
    plt.imshow(norm_data, origin='lower', cmap='gray', vmin=np.min(norm_data), vmax=np.max(norm_data))
    plt.title("Deep Template - Source Check")
    
    # Mark Target
    tx, ty = wcs.world_to_pixel(target_coord)
    plt.plot(tx, ty, 'rx', markersize=15, markeredgewidth=3, label='Target')
    
    # Mark Stars
    for i, (star, flux) in enumerate(zip(stars, fluxes)):
        sx, sy = wcs.world_to_pixel(star)
        
        # Color code: Bright/Solver vs Faint/Fixed
        if flux > 1e-4: # Arbitrary threshold for visual distinction
            color = 'cyan'
            marker = 'o'
            size = 12
        else:
            color = 'orange'
            marker = 'x'
            size = 8
            
        plt.plot(sx, sy, marker=marker, color=color, markersize=size, fillstyle='none', markeredgewidth=1.5)
        if flux > 1e-4:
            plt.text(sx+4, sy+4, f"{i}", color=color, fontsize=10, fontweight='bold')

    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(config.DIAGNOSTIC_DIR, 'PRE_ANALYSIS_CHECK.png'))
    plt.close()

def plot_fit_template_with_stars(template, wcs, stars, fluxes, target_coord):
    """
    Plots the reconstructed template model with identified stars marked.
    """
    vmin, vmax = get_robust_limits(template, 0.5, 99.5)
    
    plt.figure(figsize=(10, 10))
    # Use log scale visualization if dynamic range is large, otherwise linear
    if vmax > 0 and vmin > 0 and vmax/vmin > 100:
        norm_data = np.log10(np.clip(template, vmin, vmax))
        plt.imshow(norm_data, origin='lower', cmap='gray')
    else:
        plt.imshow(template, origin='lower', cmap='gray', vmin=vmin, vmax=vmax)
    
    # Mark Target
    tx, ty = wcs.world_to_pixel(target_coord)
    plt.plot(tx, ty, 'rx', markersize=15, markeredgewidth=3, label='Transient Loc')
    
    # Mark Stars
    for i, (star, flux) in enumerate(zip(stars, fluxes)):
        sx, sy = wcs.world_to_pixel(star)
        color = 'cyan'
        # Heuristic: Larger circle for brighter stars
        ms = 10 + np.log10(max(flux, 1e-9))*2
        ms = max(5, min(ms, 20))
        
        plt.plot(sx, sy, 'o', color=color, markersize=ms, fillstyle='none', markeredgewidth=2)
        plt.text(sx+3, sy+3, f"{i}", color=color, fontsize=12, fontweight='bold')
        
    plt.title("Reconstructed Fit Template (Scene+Stars)")
    plt.legend()
    plt.savefig(os.path.join(config.DIAGNOSTIC_DIR, 'DIAGNOSTIC_FIT_TEMPLATE.png'))
    plt.close()

def plot_bcd_residuals(cutouts, results, stars, star_fluxes, deep_stack):
    """
    Plots a 9-panel row for 5 Science BCDs and 5 Template BCDs.
    """
    print("   [Diagnostics] Generating Detailed BCD Residuals...")
    
    # Separate Science and Template indices
    sci_indices = [i for i, c in enumerate(cutouts) if not c.get('is_template', False)]
    tpl_indices = [i for i, c in enumerate(cutouts) if c.get('is_template', False)]
    
    # Randomly select 5 of each (or all if < 5)
    rng = np.random.default_rng(42)
    sel_sci = rng.choice(sci_indices, size=min(5, len(sci_indices)), replace=False)
    sel_tpl = rng.choice(tpl_indices, size=min(5, len(tpl_indices)), replace=False)
    
    # Combine and sort by epoch to keep order sensible
    selection = np.sort(np.concatenate([sel_sci, sel_tpl]))
    
    model_scene = results['model_scene']
    epoch_bgs = results['epoch_backgrounds']
    transient_fluxes = results['transient_fluxes']
    
    # For Star-Only Model, we assume model_scene IS the star-only model (plus extended if any)
    # If model_scene contains galaxy, we'd need to separate it. Assuming here model_scene = Stars.
    star_model = model_scene
    
    pdf_path = os.path.join(config.DIAGNOSTIC_DIR, 'DIAGNOSTIC_DETAILED_RESIDUALS.pdf')
    
    with PdfPages(pdf_path) as pdf:
        for k in selection:
            c = cutouts[k]
            bg = epoch_bgs[c['epoch_id']]
            t_flux = transient_fluxes[k]
            
            # --- MODELS ---
            data = c['data']
            
            # 1. Deep Stack (passed in args)
            # 2. Model No Trans = Scene + BG
            mod_no_trans = star_model + bg
            
            # 3. Model With Trans = Scene + BG + Transient (Approx point source at center)
            # We add a simple Gaussian or the PRF at center for visualization
            # Visual approx: add flux to center pixel region
            h, w = data.shape
            mod_with_trans = mod_no_trans.copy()
            # Ideally we use the real PRF, but for a quick plot, a blob suffices if PRF not avail locally
            # But we want to see if the FIT worked.
            # Note: We don't have the transient PRF shape here easily without importing solver.
            # We will render a dot for visual indication.
            mod_with_trans[h//2, w//2] += t_flux # Very crude, but indicates flux presence
            
            # 4. Star Only
            mod_stars = star_model
            
            # --- RESIDUALS ---
            res_deep = data - deep_stack
            res_no_trans = data - mod_no_trans
            res_w_trans = data - mod_with_trans
            res_stars = data - mod_stars # Note: This will leave BG and Transient
            
            # --- PLOTTING ---
            fig, axes = plt.subplots(1, 9, figsize=(45, 5))
            
            # Common Scaling for Data/Models
            vmin, vmax = get_robust_limits(data)
            # Common Scaling for Residuals (symmetric)
            r_std = np.nanstd(res_no_trans)
            rvmin, rvmax = -3*r_std, 3*r_std
            
            # 1. BCD
            axes[0].imshow(data, origin='lower', vmin=vmin, vmax=vmax, cmap='gray')
            axes[0].set_title(f"BCD\n{c['filename'][-15:]}")
            
            # 2. Deep Stack
            axes[1].imshow(deep_stack, origin='lower', vmin=vmin, vmax=vmax, cmap='gray')
            axes[1].set_title("Deep Stack")
            
            # 3. Resid (BCD - Deep)
            axes[2].imshow(res_deep, origin='lower', vmin=rvmin, vmax=rvmax, cmap='RdBu_r')
            axes[2].set_title("Resid (Deep)")
            
            # 4. Model (No Trans)
            axes[3].imshow(mod_no_trans, origin='lower', vmin=vmin, vmax=vmax, cmap='gray')
            axes[3].set_title("Model (No Trans)")
            
            # 5. Resid (No Trans)
            axes[4].imshow(res_no_trans, origin='lower', vmin=rvmin, vmax=rvmax, cmap='RdBu_r')
            axes[4].set_title("Resid (No Trans)")
            
            # 6. Model (W/ Trans)
            axes[5].imshow(mod_with_trans, origin='lower', vmin=vmin, vmax=vmax, cmap='gray')
            axes[5].set_title(f"Model (+Trans)\nFlux={t_flux:.1e}")
            
            # 7. Resid (W/ Trans)
            axes[6].imshow(res_w_trans, origin='lower', vmin=rvmin, vmax=rvmax, cmap='RdBu_r')
            axes[6].set_title("Resid (+Trans)")
            
            # 8. Star Only Model
            axes[7].imshow(mod_stars, origin='lower', vmin=vmin, vmax=vmax, cmap='gray')
            axes[7].set_title("Star Model Only")
            
            # 9. Resid (Star Only)
            axes[8].imshow(res_stars, origin='lower', vmin=vmin, vmax=vmax, cmap='RdBu_r') # Data scale for this one as it contains BG
            axes[8].set_title("Resid (Stars Only)")
            
            for ax in axes: ax.axis('off')
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close()

def plot_epoch_stacks(cutouts, results, target_coord):
    """
    Plots the median residual image for each epoch on SEPARATE pages.
    """
    print("   [Diagnostics] Generating Epoch Stacks PDF...")
    pdf_path = os.path.join(config.DIAGNOSTIC_DIR, 'DIAGNOSTIC_EPOCH_STACKS.pdf')
    
    epoch_ids = np.array([c['epoch_id'] for c in cutouts])
    unique_epochs = np.unique(epoch_ids)
    
    model_scene = results['model_scene']
    epoch_bgs = results['epoch_backgrounds']
    
    with PdfPages(pdf_path) as pdf:
        for ep in unique_epochs:
            mask = (epoch_ids == ep)
            subset_indices = np.where(mask)[0]
            
            stack_resid = []
            for idx in subset_indices:
                c = cutouts[idx]
                bg = epoch_bgs[ep]
                res = c['data'] - model_scene - bg
                stack_resid.append(res)
                
            if stack_resid:
                med_stack = np.nanmedian(np.array(stack_resid), axis=0)
                
                fig = plt.figure(figsize=(8, 8))
                vmin, vmax = get_robust_limits(med_stack)
                plt.imshow(med_stack, origin='lower', vmin=vmin, vmax=vmax, cmap='RdBu_r')
                plt.colorbar(label='Flux (Jy)')
                plt.title(f"Epoch {ep} Median Residual ({len(stack_resid)} frames)")
                
                # Mark target
                h, w = med_stack.shape
                plt.plot(w/2, h/2, 'gx', markersize=12, markeredgewidth=2)
                
                plt.tight_layout()
                pdf.savefig(fig)
                plt.close()

def plot_lightcurve(table, chan_str):
    plt.figure(figsize=(12, 6))
    mask_tpl = (table['Is_Template'] == 1)
    if np.sum(mask_tpl) > 0:
        plt.errorbar(table['MJD'][mask_tpl], table['Flux_Jy'][mask_tpl], yerr=table['Flux_Err_Jy'][mask_tpl], fmt='o', color='gray', label='Template', alpha=0.5)
    mask_sci = (table['Is_Template'] == 0)
    if np.sum(mask_sci) > 0:
        plt.errorbar(table['MJD'][mask_sci], table['Flux_Jy'][mask_sci], yerr=table['Flux_Err_Jy'][mask_sci], fmt='o', color='blue', label='Science')
        
    epoch_ids = np.unique(table['Epoch_ID'])
    bin_mjd, bin_flux, bin_err = [], [], []
    for ep in epoch_ids:
        mask = (table['Epoch_ID'] == ep) & (table['Is_Template'] == 0)
        if np.sum(mask) > 0:
            fluxes = table['Flux_Jy'][mask]
            errs = table['Flux_Err_Jy'][mask]
            w = 1.0 / (errs**2 + 1e-20)
            avg = np.sum(fluxes * w) / np.sum(w)
            bin_mjd.append(np.median(table['MJD'][mask]))
            bin_flux.append(avg)
            bin_err.append(np.sqrt(1.0/np.sum(w)))
            
    if bin_mjd:
        plt.errorbar(bin_mjd, bin_flux, yerr=bin_err, fmt='s', color='red', markersize=8, markeredgecolor='k', label='Binned')
        
    plt.xlabel("MJD"); plt.ylabel("Flux (Jy)"); plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(config.OUTPUT_DIR, f'lightcurve_{chan_str}.png'))
    plt.close()
