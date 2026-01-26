"""detect_stars_standalone.py"""
import os
import sys
import numpy as np
import sep
from astropy.io import fits
from astropy.table import Table, vstack
from scipy.spatial import cKDTree
from scipy.ndimage import median_filter, binary_erosion, shift
from scipy.interpolate import interp1d
from scipy.optimize import minimize

# --- KERNELS & MASKS ---
def get_kernel():
    """Ricker Wavelet (Mexican Hat) Kernel."""
    kernel = np.array([
        [ 0, -1, -2, -1,  0],
        [-1, -2,  1, -2, -1],
        [-2,  1, 20,  1, -2],
        [-1, -2,  1, -2, -1],
        [ 0, -1, -2, -1,  0]
    ])
    return kernel / np.max(kernel)

def get_valid_mask(data, border=5):
    """Masks edges (eroded by 5px) and NaNs."""
    valid = (data != 0) & np.isfinite(data)
    eroded = binary_erosion(valid, iterations=border)
    return eroded

# --- GALAXY MODELING ---
def cost_asymmetry(coords, image_patch):
    """Minimize rotational asymmetry."""
    dy, dx = coords
    shifted = shift(image_patch, shift=[dy, dx], order=1, mode='nearest')
    rotated = np.flip(shifted)
    diff = shifted - rotated
    return np.sum(diff**2)

def find_galaxy_center_optimized(data):
    """Finds galaxy center via asymmetry minimization."""
    h, w = data.shape
    # Rough Lock
    smooth = median_filter(data, size=21)
    yc, xc = np.unravel_index(np.argmax(smooth), smooth.shape)
    
    # Fine Optimization
    r_box = 20
    y_sl = slice(max(0, yc - r_box), min(h, yc + r_box))
    x_sl = slice(max(0, xc - r_box), min(w, xc + r_box))
    
    patch = data[y_sl, x_sl].copy()
    patch -= np.min(patch)
    if np.max(patch) > 0: patch /= np.max(patch)
    
    res = minimize(cost_asymmetry, x0=[0.0, 0.0], args=(patch,),
                   method='Nelder-Mead', tol=1e-4)
    dy_opt, dx_opt = res.x
    y_fine = yc - dy_opt
    x_fine = xc - dx_opt
    return x_fine, y_fine

def get_hybrid_residual(data):
    """
    Returns (Residual Image, Galaxy Center X, Galaxy Center Y).
    Strategy: Radial Profile Subtraction -> Median Filter Polish.
    """
    h, w = data.shape
    xc, yc = find_galaxy_center_optimized(data)
    
    # 1. Radial Profile
    y, x = np.mgrid[:h, :w]
    radii = np.sqrt((x - xc)**2 + (y - yc)**2)
    max_r = int(np.max(radii))
    r_bins = np.arange(0, max_r + 1, 0.5)
    
    flat_r = radii.ravel(); flat_d = data.ravel()
    bin_idxs = np.digitize(flat_r, r_bins)
    
    profile_r = []; profile_flux = []
    for i in range(1, len(r_bins)):
        mask = (bin_idxs == i)
        if np.sum(mask) > 0:
            val = np.nanmedian(flat_d[mask])
            profile_r.append(r_bins[i-1])
            profile_flux.append(val)
            
    interp_func = interp1d(profile_r, profile_flux, kind='linear', bounds_error=False, fill_value="extrapolate")
    radial_model = interp_func(radii)
    resid_1 = data - radial_model
    
    # 2. Median Polish (Dipole Removal)
    asym_model = median_filter(resid_1, size=20)
    resid_final = resid_1 - asym_model
    
    return resid_final, xc, yc

# --- DETECTION PASSES ---
def detect_pass_field(data, kernel):
    """Pass 1: Standard detection on original image."""
    print("   [Pass 1] Field Detection...")
    # Large mesh for global gradient
    bkg = sep.Background(data, bw=64, bh=64, fw=3, fh=3)
    # Standard threshold
    objs = sep.extract(data - bkg, 1.5, err=bkg.rms(), filter_kernel=kernel)
    return objs

def detect_pass_core(data, kernel):
    """Pass 2: Sensitive detection on Hybrid Residual."""
    print("   [Pass 2] Core Detection (Hybrid Residual)...")
    
    residual, xc, yc = get_hybrid_residual(data)
    
    # MASK NUCLEUS: Zero out central 4 pixels
    h, w = data.shape
    y_g, x_g = np.mgrid[0:h, 0:w]
    dist_sq = (x_g - xc)**2 + (y_g - yc)**2
    residual[dist_sq < 16.0] = 0.0 # 4px radius mask
    
    # Save diagnostic
    fits.writeto("diagnostic_hybrid_residual.fits", residual, overwrite=True)
    
    # Adaptive local background on residual
    bkg = sep.Background(residual, bw=8, bh=8, fw=3, fh=3)
    
    # Sensitive threshold
    objs = sep.extract(residual, 1.0, err=bkg.rms(),
                       filter_kernel=kernel, deblend_cont=0.001, deblend_nthresh=32)
    return objs

# --- MERGING & FILTERING ---
def merge_detections(field_objs, core_objs):
    """
    Merges Field and Core detections.
    If a star appears in both, keep Field (measured on real data).
    Add Core stars only if unique.
    """
    if len(field_objs) == 0: return core_objs
    if len(core_objs) == 0: return field_objs
    
    # Convert Field to Tree
    field_coords = np.column_stack((field_objs['x'], field_objs['y']))
    tree = cKDTree(field_coords)
    
    # Check Core against Field
    core_coords = np.column_stack((core_objs['x'], core_objs['y']))
    dists, idxs = tree.query(core_coords)
    
    # Keep Core stars that are > 2.0 pixels away from any Field star
    unique_mask = dists > 2.0
    unique_core = core_objs[unique_mask]
    
    print(f"   Merging: {len(field_objs)} Field + {len(unique_core)} Unique Core.")
    
    # Sep objects are numpy structured arrays, can be concatenated
    return np.concatenate((field_objs, unique_core))

def filter_sources(objects, valid_mask, data_shape):
    """
    Applies global cuts: Edge, Shape, Bright Star Ghosts.
    """
    h, w = data_shape
    keep = np.ones(len(objects), dtype=bool)
    
    for i in range(len(objects)):
        x, y = int(objects['x'][i]), int(objects['y'][i])
        
        # Bounds & Mask (Edge)
        if x < 0 or x >= w or y < 0 or y >= h:
            keep[i] = False; continue
        if not valid_mask[y, x]:
            keep[i] = False; continue
            
        # Shape (Roundness)
        if (objects['b'][i] / objects['a'][i]) < 0.5:
            keep[i] = False; continue

    # Bright Star Ghosting
    fluxes = objects['flux']
    if len(fluxes) > 0:
        bright_thresh = np.percentile(fluxes, 99.5)
        bright_idxs = np.where(fluxes > bright_thresh)[0]
        for b_idx in bright_idxs:
            bx, by = objects['x'][b_idx], objects['y'][b_idx]
            dist_sq = (objects['x'] - bx)**2 + (objects['y'] - by)**2
            
            # Kill faint neighbors within 15px, but KEEP THE STAR ITSELF
            ghosts = (dist_sq < 15**2) & (np.arange(len(objects)) != b_idx)
            keep[ghosts] = False

    return objects[keep]

def update_source_catalog(new_objects, output_dir="output"):
    catalog_path = os.path.join(output_dir, "source_catalog.fits")
    
    t_new = Table()
    t_new['filename'] = ['deep_template'] * len(new_objects)
    for col in ['x', 'y', 'flux', 'a', 'b', 'theta']:
        t_new[col] = new_objects[col]
    
    if not os.path.exists(catalog_path):
        t_new.write(catalog_path, overwrite=True)
        return

    try:
        t_existing = Table.read(catalog_path)
        mask = t_existing['filename'] != 'deep_template'
        t_clean = t_existing[mask]
        t_final = vstack([t_clean, t_new])
        t_final.write(catalog_path, overwrite=True)
        print(f"   Catalog updated: {len(t_final)} rows.")
    except Exception as e:
        print(f"   Catalog update failed: {e}")

def write_regions(filename, objects):
    with open(filename, 'w') as f:
        f.write("# Region file format: DS9 version 4.1\n")
        f.write('global color=green dashlist=8 3 width=2 font="helvetica 10 normal roman" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1\n')
        f.write("image\n")
        for i in range(len(objects)):
            x, y = objects['x'][i] + 1, objects['y'][i] + 1
            f.write(f"circle({x:.3f},{y:.3f},3.0)\n")
    print(f"   Regions saved to {filename}")

def run_detection(fits_path):
    print(f"Loading {fits_path}...")
    if not os.path.exists(fits_path): return

    with fits.open(fits_path) as hdul:
        data = hdul[0].data
    data = np.nan_to_num(data).astype(float)
    if not data.flags['C_CONTIGUOUS']: data = np.ascontiguousarray(data)
    
    kernel = get_kernel()
    valid_mask = get_valid_mask(data, border=5) # 5px edge erosion
    
    # 1. Run Passes
    objs_field = detect_pass_field(data, kernel)
    objs_core = detect_pass_core(data, kernel)
    
    # 2. Merge
    objs_merged = merge_detections(objs_field, objs_core)
    
    # 3. Filter
    objs_clean = filter_sources(objs_merged, valid_mask, data.shape)
    print(f"   Final Count: {len(objs_clean)} (Filtered from {len(objs_merged)})")
    
    # 4. Output
    write_regions("all_detected_stars.reg", objs_clean)
    update_source_catalog(objs_clean)
    print("\nDone.")

if __name__ == "__main__":
    default_path = os.path.join("output", "deep_template_ch2.fits")
    target_file = sys.argv[1] if len(sys.argv) > 1 else default_path
    run_detection(target_file)
