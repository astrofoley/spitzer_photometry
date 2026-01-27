"""src/solver.py"""
import sys
import os
import glob
import numpy as np
from scipy.linalg import lstsq
from scipy.ndimage import gaussian_filter
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from reproject import reproject_interp
from . import config, gp_model
from .pmap_correction import iracpc_pmap_corr

# ... [PRF Utilities: GRID_CENTERS, get_gaussian_kernel_array, apply_window_function, load_real_prf_file, load_interpolated_prf, load_prf unchanged] ...

GRID_CENTERS = np.array([25, 77, 129, 181, 233])

def get_gaussian_kernel_array(size=2001, fwhm=2.0, oversample=100):
    y, x = np.mgrid[0:size, 0:size]
    cy, cx = size // 2, size // 2
    sigma = (fwhm * oversample) / 2.355
    r2 = (x - cx)**2 + (y - cy)**2
    kernel = np.exp(-r2 / (2 * sigma**2))
    return kernel / np.sum(kernel)

def apply_window_function(data):
    h, w = data.shape
    alpha = 0.1
    x = np.linspace(0, 1, w); y = np.linspace(0, 1, h)
    wx = np.ones_like(x); mask_x = (x < alpha/2)
    wx[mask_x] = 0.5 * (1 + np.cos(2*np.pi/alpha * (x[mask_x] - alpha/2)))
    mask_x = (x > 1 - alpha/2)
    wx[mask_x] = 0.5 * (1 + np.cos(2*np.pi/alpha * (x[mask_x] - 1 + alpha/2)))
    wy = np.ones_like(y); mask_y = (y < alpha/2)
    wy[mask_y] = 0.5 * (1 + np.cos(2*np.pi/alpha * (y[mask_y] - alpha/2)))
    mask_y = (y > 1 - alpha/2)
    wy[mask_y] = 0.5 * (1 + np.cos(2*np.pi/alpha * (y[mask_y] - 1 + alpha/2)))
    return data * np.outer(wy, wx)

def load_real_prf_file(channel, row_idx, col_idx):
    if not (0 <= row_idx < 5 and 0 <= col_idx < 5): return None
    r_val = GRID_CENTERS[row_idx]; c_val = GRID_CENTERS[col_idx]
    ch_num = '2' if 'ch2' in channel.lower() else '1'
    filename = f"apex_sh_IRACPC{ch_num}_col{c_val:03d}_row{r_val:03d}_x100.fits"
    patterns = [filename, f"prf_{channel}_row{row_idx}_col{col_idx}.fits"]
    for pat in patterns:
        path = os.path.join(config.PRF_DIR, pat)
        if os.path.exists(path):
            try:
                with fits.open(path) as hdul:
                    data = np.nan_to_num(hdul[0].data, nan=0.0)
                    return apply_window_function(data)
            except: continue
    return None

def load_interpolated_prf(channel, x_det, y_det):
    grid_centers = GRID_CENTERS
    x = np.clip(x_det, grid_centers[0], grid_centers[-1])
    y = np.clip(y_det, grid_centers[0], grid_centers[-1])
    ix = np.searchsorted(grid_centers, x, side='right'); iy = np.searchsorted(grid_centers, y, side='right')
    ix = max(1, min(ix, 4)); iy = max(1, min(iy, 4))
    x0, x1 = grid_centers[ix-1], grid_centers[ix]
    y0, y1 = grid_centers[iy-1], grid_centers[iy]
    u = (x - x0) / (x1 - x0); v = (y - y0) / (y1 - y0)
    p00 = load_real_prf_file(channel, iy-1, ix-1)
    p10 = load_real_prf_file(channel, iy-1, ix)
    p01 = load_real_prf_file(channel, iy, ix-1)
    p11 = load_real_prf_file(channel, iy, ix)
    if all(p is None for p in [p00, p10, p01, p11]):
        return get_gaussian_kernel_array(size=2001, fwhm=2.0, oversample=config.PRF_OVERSAMPLE)
    if p00 is None: p00 = p10 if p10 else (p01 if p01 else p11)
    if p10 is None: p10 = p00;
    if p01 is None: p01 = p00;
    if p11 is None: p11 = p10
    w00 = (1-u)*(1-v); w10 = u*(1-v); w01 = (1-u)*v; w11 = u*v
    eff_prf = w00*p00 + w10*p10 + w01*p01 + w11*p11
    if np.sum(eff_prf) > 0: eff_prf /= np.sum(eff_prf)
    return eff_prf

def load_prf(channel, x_det=None, y_det=None):
    if x_det is None: x_det = 128
    if y_det is None: y_det = 128
    return load_interpolated_prf(channel, x_det, y_det)

# --- Core Solver ---

DEBUG_DUMPED = False

def generate_prf_fast(scene_wcs, raw_wcs, prf_model, ra, dec, scene_shape, channel='ch1', is_full_array=False):
    global DEBUG_DUMPED
    if prf_model is None: return np.zeros(scene_shape[0]*scene_shape[1])
    
    ph, pw = prf_model.shape
    prf_wcs = WCS(naxis=2)
    prf_wcs.wcs.crpix = [pw / 2.0, ph / 2.0]
    prf_wcs.wcs.crval = [ra, dec]
    prf_wcs.wcs.ctype = raw_wcs.wcs.ctype
    
    if hasattr(raw_wcs.wcs, 'cd'):
        prf_wcs.wcs.cd = raw_wcs.wcs.cd / float(config.PRF_OVERSAMPLE)
    else:
        prf_wcs.wcs.cdelt = raw_wcs.wcs.cdelt / float(config.PRF_OVERSAMPLE)
        prf_wcs.wcs.pc = raw_wcs.wcs.pc
    
    zoom_factor = float(config.SUPERSAMPLE_FACTOR) / float(config.PRF_OVERSAMPLE)
    sigma = 0.5 * (1.0 / zoom_factor)
    prf_smooth = gaussian_filter(prf_model, sigma)
    
    try:
        out_arr, _ = reproject_interp((prf_smooth, prf_wcs), scene_wcs, shape_out=scene_shape)
        out_arr = np.nan_to_num(out_arr)
    except:
        return np.zeros(scene_shape[0]*scene_shape[1])
    
    curr_sum = np.sum(out_arr)
    if curr_sum > 0:
        out_arr /= curr_sum
        
    # --- Intrapixel Correction ---
    tx, ty = raw_wcs.world_to_pixel_values(ra, dec)
    
    try:
        # Strict usage: Pass Full Array flag, no periodic logic.
        corr_val = iracpc_pmap_corr(1.0, tx, ty, channel,
                                    pmap_dir='data/pmap_fits',
                                    threshold_occ=False,
                                    full_array=is_full_array)
        
        if np.isfinite(corr_val) and corr_val > 0:
            gain = 1.0 / corr_val
            out_arr *= gain
            gain_msg = f"Gain={gain:.4f}"
        else:
            gain_msg = "Gain=1.0 (OOB/Masked)"
    except Exception as e:
        print(f"   [PMap Warning] Failed for ({tx:.2f}, {ty:.2f}): {e}")
        gain_msg = "Gain=1.0 (Error)"
        
    if not DEBUG_DUMPED:
        out_path = os.path.join(config.DIAGNOSTIC_DIR, 'DEBUG_PRF_GENERATED.fits')
        fits.writeto(out_path, out_arr, overwrite=True)
        print(f"   [DEBUG PRF] Loc: ({tx:.2f}, {ty:.2f}) [Full={is_full_array}] -> {gain_msg}")
        DEBUG_DUMPED = True

    return out_arr.flatten()

def run_gls_solve(cutouts, stars, star_initial_fluxes, gp_params, regularization, deep_template, template_wcs, n_epochs):
    global DEBUG_DUMPED
    DEBUG_DUMPED = False
    
    if not cutouts: return None
    scene_shape = (cutouts[0]['data'].shape[0], cutouts[0]['data'].shape[1])
    n_scene = scene_shape[0]*scene_shape[1]
    
    results = {
        'transient_fluxes': np.zeros(len(cutouts)),
        'transient_errs': np.zeros(len(cutouts)),
        'star_fluxes': np.zeros(len(stars)),
        'epoch_backgrounds': np.zeros(n_epochs),
        'model_scene': np.zeros(scene_shape),
        'scene_wcs': None,
        'scene_shape': scene_shape
    }

    try:
        print("   [Solver] Step 1: Geometry Setup...")
        sys.stdout.flush()
        
        ell, var = regularization
        target_loc = SkyCoord(config.TRANSIENT_RA, config.TRANSIENT_DEC, unit='deg')
        
        scene_wcs = WCS(naxis=2)
        scene_wcs.wcs.crpix = [scene_shape[1]/2, scene_shape[0]/2]
        scene_wcs.wcs.crval = [target_loc.ra.deg, target_loc.dec.deg]
        scene_wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
        scale = config.PIXEL_SCALE / config.SUPERSAMPLE_FACTOR / 3600.0
        scene_wcs.wcs.cdelt = [-scale, scale]
        scene_wcs.wcs.pc = np.eye(2)
        
        results['scene_wcs'] = scene_wcs
        
        y, x = np.mgrid[0:scene_shape[0], 0:scene_shape[1]]
        coords = np.vstack([y.ravel(), x.ravel()]).T
        Q = gp_model.matern32_kernel(coords, ell * config.SUPERSAMPLE_FACTOR, var)
        Q += np.eye(n_scene) * 1e-6
        Q_inv = np.linalg.inv(Q)
        
        solver_stars = []
        solver_init = []
        full_map_indices = []
        
        for i, (s, f) in enumerate(zip(stars, star_initial_fluxes)):
            if s.separation(target_loc).arcsec > 1.5:
                solver_stars.append(s)
                solver_init.append(f)
                full_map_indices.append(i)
                
        n_stars = len(solver_stars)
        n_trans = len(cutouts)
        n_bg = n_epochs
        n_params = n_scene + n_trans + n_stars + n_bg
        
        H = np.zeros((n_params, n_params))
        rhs = np.zeros(n_params)
        
        H[:n_scene, :n_scene] = Q_inv
        
        idx_trans = n_scene
        idx_stars = idx_trans + n_trans
        idx_bg = idx_stars + n_stars
        
        print(f"   [Solver] Step 2: Matrix Fill ({n_params} params)...")
        sys.stdout.flush()
        
        for i, entry in enumerate(cutouts):
            d = entry['data'].flatten()
            s = entry['sigma'].flatten()
            mask = (entry['data'] != 0) & np.isfinite(entry['sigma'])
            if np.sum(mask) == 0: continue
            
            w = np.zeros(n_scene)
            w[mask.flatten()] = 1.0 / (np.clip(s[mask.flatten()], 1e-9, None)**2)
            
            np.fill_diagonal(H[:n_scene, :n_scene], np.diag(H[:n_scene, :n_scene]) + w)
            rhs[:n_scene] += w * d
            
            chan = 'ch2' if 'ch2' in entry['filename'] else 'ch1'
            w_native = entry['raw_wcs']
            
            # --- Retrieve Full Array Flag ---
            is_full = entry.get('is_full_array', False)
            
            tx, ty = w_native.world_to_pixel_values(config.TRANSIENT_RA, config.TRANSIENT_DEC)
            prf_t = load_prf(chan, tx, ty)
            
            # --- Pass Flag to Generator ---
            col_t = generate_prf_fast(scene_wcs, w_native, prf_t, config.TRANSIENT_RA, config.TRANSIENT_DEC, scene_shape,
                                      channel=chan, is_full_array=is_full)
            
            it = idx_trans + i
            H[it, it] += np.dot(col_t, col_t * w)
            rhs[it] += np.dot(col_t, w * d)
            H[:n_scene, it] += col_t * w
            H[it, :n_scene] += col_t * w
            
            cols_s = []
            for s_obj in solver_stars:
                sx, sy = w_native.world_to_pixel_values(s_obj.ra.deg, s_obj.dec.deg)
                prf_s = load_prf(chan, sx, sy)
                # --- Pass Flag to Generator ---
                cols_s.append(generate_prf_fast(scene_wcs, w_native, prf_s, s_obj.ra.deg, s_obj.dec.deg, scene_shape,
                                                channel=chan, is_full_array=is_full))
            
            if cols_s:
                S = np.column_stack(cols_s)
                H[idx_stars:idx_bg, idx_stars:idx_bg] += S.T @ (S * w[:, None])
                rhs[idx_stars:idx_bg] += S.T @ (w * d)
                H[:n_scene, idx_stars:idx_bg] += S * w[:, None]
                H[idx_stars:idx_bg, :n_scene] += (S * w[:, None]).T
                H[it, idx_stars:idx_bg] += col_t @ (S * w[:, None])
                H[idx_stars:idx_bg, it] += (S * w[:, None]).T @ col_t

            ib = idx_bg + entry['epoch_id']
            H[ib, ib] += np.sum(w)
            rhs[ib] += np.dot(w, d)
            H[:n_scene, ib] += w
            H[ib, :n_scene] += w
            
            if cols_s:
                H[idx_stars:idx_bg, ib] += np.sum(S * w[:, None], axis=0)
                H[ib, idx_stars:idx_bg] += np.sum(S * w[:, None], axis=0)
            H[it, ib] += np.sum(col_t * w)
            H[ib, it] += np.sum(col_t * w)

        print("   [Solver] Step 3: Prior...")
        sys.stdout.flush()
        if n_stars > 0:
            star_diag = np.diag(H[idx_stars:idx_bg, idx_stars:idx_bg])
            avg_w = np.median(star_diag)
            lam = max((avg_w / len(cutouts)) * 5.0, 1.0)
            
            for k, f_init in enumerate(solver_init):
                im = idx_stars + k
                H[im, im] += lam
                rhs[im] += lam * f_init

        for i, entry in enumerate(cutouts):
            if entry.get('is_template'):
                it = idx_trans + i
                H[it, :] = 0; H[:, it] = 0; H[it, it] = 1; rhs[it] = 0

        print("   [Solver] Step 4: LSTSQ...")
        sys.stdout.flush()
        sol = lstsq(H, rhs, check_finite=False, lapack_driver='gelsy')[0]
        
        results['model_scene'] = sol[:n_scene].reshape(scene_shape)
        results['transient_fluxes'] = sol[idx_trans:idx_stars]
        
        fitted_star_fluxes = sol[idx_stars:idx_bg]
        full_fluxes = np.zeros(len(stars))
        for k, valid_idx in enumerate(full_map_indices):
            full_fluxes[valid_idx] = fitted_star_fluxes[k]
        results['star_fluxes'] = full_fluxes
        
        results['epoch_backgrounds'] = sol[idx_bg:]
        
    except Exception as e:
        print(f"CRITICAL SOLVER ERROR: {e}")
        import traceback
        traceback.print_exc()
    
    return results

def reconstruct_full_model(results, star_coords, star_fluxes, config_channel, cutouts):
    scene_model = results['model_scene'].copy()
    scene_wcs = results['scene_wcs']
    if scene_wcs is None: return scene_model
    
    scene_shape = results['scene_shape']
    n = min(len(star_fluxes), len(star_coords))
    
    chan = 'ch2' if 'ch2' in config_channel else 'ch1'
    accum = np.zeros_like(scene_model)
    counts = 0
    stride = max(1, len(cutouts) // 50)
    
    for i in range(0, len(cutouts), stride):
        w_native = cutouts[i]['raw_wcs']
        is_full = cutouts[i].get('is_full_array', False)
        
        field = np.zeros(scene_shape[0]*scene_shape[1])
        for j in range(n):
            if star_fluxes[j] <= 0: continue
            sx, sy = w_native.world_to_pixel_values(star_coords[j].ra.deg, star_coords[j].dec.deg)
            prf = load_prf(chan, sx, sy)
            field += generate_prf_fast(scene_wcs, w_native, prf, star_coords[j].ra.deg, star_coords[j].dec.deg, scene_shape,
                                       channel=chan, is_full_array=is_full) * star_fluxes[j]
        accum += field.reshape(scene_shape)
        counts += 1
        
    if counts > 0: scene_model += accum / counts
    return scene_model
