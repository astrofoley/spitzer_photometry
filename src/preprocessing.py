"""src/preprocessing.py"""
import os
import glob
import warnings
import numpy as np
import sep
from astropy.io import fits
from astropy.table import Table, vstack, Column
from astropy.wcs import WCS
from astropy.time import Time
from astropy.coordinates import SkyCoord
from scipy.spatial import cKDTree
from scipy.ndimage import median_filter, binary_erosion, shift
from scipy.interpolate import interp1d
from scipy.optimize import minimize
from reproject import reproject_interp
from . import config

# --- HELPER FUNCTIONS ---

def parse_mjd_robust(header):
    for key in ['MJD-OBS', 'MJD_OBS', 'MJD']:
        if key in header: return float(header[key])
    for key in ['DATE-OBS', 'DATE_OBS', 'DATE']:
        if key in header:
            try: return Time(header[key]).mjd
            except: continue
    return None

def is_correct_channel(filename, config_channel):
    fname = os.path.basename(filename).lower()
    chan = config_channel.lower()
    if chan in fname: return True
    if chan == 'ch1' and ('i1' in fname or 'I1' in filename): return True
    if chan == 'ch2' and ('i2' in fname or 'I2' in filename): return True
    return False

def get_valid_mask(data, border=5):
    valid = (data != 0) & np.isfinite(data)
    eroded = binary_erosion(valid, iterations=border)
    return eroded

def get_ricker_kernel():
    kernel = np.array([
        [ 0, -1, -2, -1,  0],
        [-1, -2,  1, -2, -1],
        [-2,  1, 20,  1, -2],
        [-1, -2,  1, -2, -1],
        [ 0, -1, -2, -1,  0]
    ])
    return kernel / np.max(kernel)

# --- GALAXY MODELING ---

def cost_asymmetry(coords, image_patch):
    dy, dx = coords
    shifted = shift(image_patch, shift=[dy, dx], order=1, mode='nearest')
    rotated = np.flip(shifted)
    diff = shifted - rotated
    return np.sum(diff**2)

def find_galaxy_center_optimized(data):
    h, w = data.shape
    smooth = median_filter(data, size=21)
    yc, xc = np.unravel_index(np.argmax(smooth), smooth.shape)
    r_box = 20
    y_sl = slice(max(0, yc - r_box), min(h, yc + r_box))
    x_sl = slice(max(0, xc - r_box), min(w, xc + r_box))
    patch = data[y_sl, x_sl].copy()
    patch -= np.min(patch)
    if np.max(patch) > 0: patch /= np.max(patch)
    res = minimize(cost_asymmetry, x0=[0.0, 0.0], args=(patch,), method='Nelder-Mead', tol=1e-4)
    dy_opt, dx_opt = res.x
    return xc - dx_opt, yc - dy_opt

def subtract_radial_profile(data, xc, yc):
    h, w = data.shape
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
    return interp_func(radii)

# --- DETECTION & FILTERING ---

def detect_sources(data, wcs, sigma_map=None, is_template=False, mask_nucleus=True):
    h, w = data.shape
    kernel = get_ricker_kernel()
    valid_mask = get_valid_mask(data, border=5)
    xc, yc = 0, 0
    if mask_nucleus or is_template:
        xc, yc = find_galaxy_center_optimized(data)
    
    if is_template:
        radial_model = subtract_radial_profile(data, xc, yc)
        resid_1 = data - radial_model
        asym_model = median_filter(resid_1, size=20)
        detection_image = resid_1 - asym_model
        bkg = sep.Background(detection_image, bw=32, bh=32, fw=3, fh=3)
        err = sigma_map if (sigma_map is not None and np.nanmedian(sigma_map) > 0) else bkg.rms()
        thresh = 1.5
    else:
        bkg = sep.Background(data, bw=32, bh=32, fw=3, fh=3)
        detection_image = data - bkg
        err = sigma_map if (sigma_map is not None and np.nanmedian(sigma_map) > 0) else bkg.rms()
        thresh = 3.0

    try:
        objects = sep.extract(detection_image, thresh, err=err,
                              filter_kernel=kernel, minarea=3,
                              deblend_cont=0.005, deblend_nthresh=32)
    except Exception as e:
        print(f"   SEP Extraction failed: {e}")
        return None

    if len(objects) == 0: return None

    keep = np.ones(len(objects), dtype=bool)
    transient_pix = None
    if not is_template and wcs is not None:
        try:
            tx, ty = wcs.wcs_world2pix(config.TRANSIENT_RA, config.TRANSIENT_DEC, 0)
            transient_pix = (float(tx), float(ty))
        except: pass

    for i in range(len(objects)):
        ox, oy = objects['x'][i], objects['y'][i]
        ix, iy = int(ox), int(oy)
        if ix < 0 or ix >= w or iy < 0 or iy >= h: keep[i] = False; continue
        if not valid_mask[iy, ix]: keep[i] = False; continue
        if objects['b'][i] / objects['a'][i] < 0.5: keep[i] = False; continue
        if mask_nucleus:
            if np.sqrt((ox - xc)**2 + (oy - yc)**2) < 10.0: keep[i] = False; continue
        if transient_pix:
            if np.sqrt((ox - transient_pix[0])**2 + (oy - transient_pix[1])**2) < 15.0: keep[i] = False; continue
        if sigma_map is not None:
            if sigma_map[iy, ix] > 1e10: keep[i] = False; continue

    filtered_objs = objects[keep]
    if len(filtered_objs) == 0: return None
    
    t = Table()
    t['x'] = filtered_objs['x']; t['y'] = filtered_objs['y']
    t['flux'] = filtered_objs['flux']; t['peak'] = filtered_objs['peak']
    t['a'] = filtered_objs['a']; t['b'] = filtered_objs['b']; t['theta'] = filtered_objs['theta']
    if wcs is not None:
        ra, dec = wcs.pixel_to_world_values(t['x'], t['y'])
        t['ra'] = ra; t['dec'] = dec
    return t

# --- DATA LOADING ---

def load_data(file_pair):
    img_path = file_pair['image']
    unc_path = file_pair['unc']
    if not os.path.exists(unc_path):
        raise FileNotFoundError(f"CRITICAL: Missing uncertainty file {unc_path}.")
    with fits.open(img_path) as hdul:
        data = hdul[0].data
        header = hdul[0].header
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            wcs = WCS(header)
    with fits.open(unc_path) as hdul:
        unc_data = hdul[0].data
    data = np.nan_to_num(data, nan=0.0)
    unc_data = np.nan_to_num(unc_data, nan=np.inf)
    return {'data': data, 'sigma': unc_data, 'wcs': wcs, 'header': header, 'filename': os.path.basename(img_path), 'mjd': parse_mjd_robust(header)}

def generate_sources_for_list(file_list, is_template_run=False):
    tables = []
    print(f"   Detecting sources for {len(file_list)} files...")
    for f_info in file_list:
        try:
            raw = load_data(f_info)
            data = np.ascontiguousarray(raw['data'].astype(float))
            sigma = np.ascontiguousarray(raw['sigma'].astype(float))
            t = detect_sources(data, raw['wcs'], sigma_map=sigma, is_template=is_template_run)
            if t is not None and len(t) > 0:
                t.add_column(Column([raw['filename']] * len(t), name='filename', dtype='U'))
                cols = ['filename'] + [c for c in t.colnames if c != 'filename']
                t = t[cols]
                tables.append(t)
        except Exception as e:
            continue
    if tables: return vstack(tables)
    return None

def get_or_create_source_catalog(file_list):
    catalog_path = config.SOURCE_CATALOG_PATH
    catalog = None
    existing_files = set()
    if os.path.exists(catalog_path):
        print(f"Loading source catalog from {catalog_path}...")
        try:
            catalog = Table.read(catalog_path)
            if catalog['filename'].dtype.kind in ('S', 'a'):
                catalog['filename'] = catalog['filename'].astype('U')
            existing_files = set(catalog['filename'])
            print(f"   Catalog contains {len(existing_files)} unique files.")
        except Exception as e:
            catalog = None
    missing_files = [f for f in file_list if os.path.basename(f['image']) not in existing_files]
    if missing_files:
        print(f"   Backfilling {len(missing_files)} missing science frames...")
        new_chunk = generate_sources_for_list(missing_files, is_template_run=False)
        if new_chunk:
            if catalog is None: catalog = new_chunk
            else: catalog = vstack([catalog, new_chunk])
            catalog.write(catalog_path, overwrite=True)
            print(f"   Catalog updated. Total rows: {len(catalog)}")
    if catalog is not None:
        grouped = catalog.group_by('filename')
        cat_dict = {}
        for group in grouped.groups:
            fname = str(group['filename'][0])
            cat_dict[fname] = group
        return cat_dict
    return {}

def update_catalog_with_template(deep_template, wcs, template_name='deep_template'):
    print("Updating Catalog with Deep Template sources...")
    t = detect_sources(deep_template, wcs, sigma_map=None, is_template=True, mask_nucleus=True)
    if t is not None:
        t.add_column(Column([template_name] * len(t), name='filename', dtype='U'))
        cols = ['filename'] + [c for c in t.colnames if c != 'filename']
        t = t[cols]
        catalog_path = config.SOURCE_CATALOG_PATH
        if os.path.exists(catalog_path):
            curr = Table.read(catalog_path)
            if curr['filename'].dtype.kind in ('S', 'a'): curr['filename'] = curr['filename'].astype('U')
            curr = curr[curr['filename'] != template_name]
            final = vstack([curr, t])
        else:
            final = t
        final.write(catalog_path, overwrite=True)
        print(f"   Deep Template updated: {len(t)} sources.")
        return t
    return None

def find_spitzer_files(root_dir):
    file_list = []
    cbcd_files = glob.glob(os.path.join(root_dir, '**', '*cbcd.fits'), recursive=True)
    for img_path in cbcd_files:
        if not is_correct_channel(img_path, config.CHANNEL): continue
        unc_path = img_path.replace('cbcd.fits', 'cbunc.fits')
        file_list.append({'image': img_path, 'unc': unc_path, 'filename': os.path.basename(img_path)})
    return file_list

def filter_off_chip_frames(file_list, target_ra, target_dec):
    valid_files = []
    buffer = 15
    for i, f in enumerate(file_list):
        try:
            with fits.open(f['image']) as hdul:
                header = hdul[0].header
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    w = WCS(header)
                    x, y = w.wcs_world2pix(target_ra, target_dec, 0)
                naxis1 = header.get('NAXIS1', 256)
                naxis2 = header.get('NAXIS2', 256)
            x, y = float(x), float(y)
            if (x > buffer) and (x < naxis1 - buffer) and (y > buffer) and (y < naxis2 - buffer):
                valid_files.append(f)
        except: continue
    print(f"   On-Chip Filter: {len(valid_files)}/{len(file_list)} kept.")
    return valid_files

def categorize_observations(file_pairs, split_date_mjd):
    valid_dated_pairs = []
    for pair in file_pairs:
        try:
            with fits.open(pair['image']) as hdul:
                obs_mjd = parse_mjd_robust(hdul[0].header)
            if obs_mjd is not None:
                pair['mjd'] = obs_mjd
                valid_dated_pairs.append(pair)
        except: pass
    on_target_pairs = filter_off_chip_frames(valid_dated_pairs, config.TRANSIENT_RA, config.TRANSIENT_DEC)
    science = [p for p in on_target_pairs if p['mjd'] <= split_date_mjd]
    template = [p for p in on_target_pairs if p['mjd'] > split_date_mjd]
    return science, template

def define_mosaic_wcs(file_list, target_coord):
    print("Defining Global Mosaic WCS...")
    raw0 = load_data(file_list[0])
    h_native, w_native = raw0['data'].shape
    factor = config.SUPERSAMPLE_FACTOR
    n_h, n_w = int(h_native * factor), int(w_native * factor)
    scale = config.PIXEL_SCALE / factor / 3600.0
    w_mosaic = WCS(naxis=2)
    w_mosaic.wcs.crpix = [n_w/2, n_h/2]
    w_mosaic.wcs.crval = [target_coord.ra.deg, target_coord.dec.deg]
    w_mosaic.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w_mosaic.wcs.cdelt = [-scale, scale]
    w_mosaic.wcs.pc = np.eye(2)
    return w_mosaic, (n_h, n_w)

def reproject_to_grid(file_list, target_wcs, shape):
    processed = []
    h, w = shape
    print(f"   Reprojecting {len(file_list)} frames...")
    for f in file_list:
        raw = load_data(f)
        d_jy = raw['data'] * config.MJY_SR_TO_JY
        s_jy = raw['sigma'] * config.MJY_SR_TO_JY
        wcs_use = f.get('corrected_wcs', raw['wcs'])
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            d_proj, _ = reproject_interp((d_jy, wcs_use), target_wcs, shape_out=(h, w))
            s_proj, _ = reproject_interp((s_jy, wcs_use), target_wcs, shape_out=(h, w))
        d_proj = np.nan_to_num(d_proj)
        s_proj = np.nan_to_num(s_proj, nan=np.inf)
        s_proj[(d_proj == 0) & (s_proj == 0)] = np.inf
        processed.append({'data': d_proj, 'sigma': s_proj, 'raw_wcs': raw['wcs'], 'corrected_wcs': wcs_use, 'filename': raw['filename'], 'mjd': raw['mjd'], 'file_info': f})
    return processed

def create_median_stack(data_cube):
    cube = data_cube.copy()
    cube[cube == 0] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        median = np.nanmedian(cube, axis=0)
        mad = np.nanmedian(np.abs(cube - median), axis=0)
    return np.nan_to_num(median), np.nan_to_num(mad)

def align_frames_to_template(file_list, deep_template, template_wcs, source_catalog):
    print("Refining Astrometry...")
    if 'deep_template' in source_catalog:
        objs_ref = source_catalog['deep_template']
        ref_world = template_wcs.pixel_to_world_values(objs_ref['x'], objs_ref['y'])
    else:
        print("   WARNING: Deep Template sources missing.")
        return
    if len(ref_world[0]) < 3: return
    ref_world_arr = np.column_stack((ref_world[0], ref_world[1]))
    ref_tree = cKDTree(ref_world_arr)
    aligned_count = 0
    for f in file_list:
        fname = f['filename']
        if fname not in source_catalog: continue
        src = source_catalog[fname]
        xs, ys = src['x'], src['y']
        if len(xs) < 2: continue
        try:
            with fits.open(f['image']) as hdul: wcs_orig = WCS(hdul[0].header)
        except: continue
        ra, dec = wcs_orig.pixel_to_world_values(xs, ys)
        bcd_world = np.column_stack((ra, dec))
        dists, indices = ref_tree.query(bcd_world, distance_upper_bound=0.003)
        valid = dists != float('inf')
        if np.sum(valid) < 2: continue
        matched_ref_world = ref_world_arr[indices[valid]]
        ref_px, ref_py = wcs_orig.world_to_pixel_values(matched_ref_world[:, 0], matched_ref_world[:, 1])
        matched_bcd_pix = np.column_stack((xs[valid], ys[valid]))
        matched_ref_pix = np.column_stack((ref_px, ref_py))
        offsets = matched_bcd_pix - matched_ref_pix
        dx = np.median(offsets[:, 0])
        dy = np.median(offsets[:, 1])
        wcs_corr = wcs_orig.deepcopy()
        wcs_corr.wcs.crpix[0] += dx
        wcs_corr.wcs.crpix[1] += dy
        f['corrected_wcs'] = wcs_corr
        aligned_count += 1
    print(f"   Aligned {aligned_count}/{len(file_list)} frames.")

def flag_cosmic_rays(processed_list, deep_template, scene_wcs, target_coord):
    print("Flagging CRs...")
    h, w = deep_template.shape
    y_g, x_g = np.mgrid[0:h, 0:w]
    scene_scale = config.PIXEL_SCALE / config.SUPERSAMPLE_FACTOR
    r_px = 3.0 / scene_scale
    mask_stars = np.zeros((h, w), dtype=bool)
    try:
        bkg = sep.Background(np.ascontiguousarray(deep_template.astype(float)))
        objs = sep.extract(deep_template - bkg, 3.0*bkg.globalrms)
        for i in range(len(objs)):
            ox, oy = objs['x'][i], objs['y'][i]
            d2 = (x_g - ox)**2 + (y_g - oy)**2
            mask_stars |= (d2 < r_px**2)
    except: pass
    tx, ty = scene_wcs.world_to_pixel(target_coord)
    dist_sq = (x_g - tx)**2 + (y_g - ty)**2
    mask_transient = (dist_sq < r_px**2)
    count = 0
    for entry in processed_list:
        data = entry['data']; sigma = entry['sigma']
        final_protection = mask_stars.copy()
        if not entry.get('is_template', False): final_protection |= mask_transient
        valid_mask = (data != 0) & (deep_template != 0)
        zodi_offset = np.nanmedian(data[valid_mask] - deep_template[valid_mask]) if np.sum(valid_mask) > 10 else 0.0
        diff = data - deep_template - zodi_offset
        with np.errstate(invalid='ignore', divide='ignore'): nsigma = np.abs(diff) / sigma
        is_cr = nsigma > 5.0
        is_cr[final_protection] = False
        sigma[is_cr] = np.inf
        entry['sigma'] = sigma
        count += np.sum(is_cr)
    print(f"   Masked {count} CR pixels.")
    return count

def extract_analysis_cutouts(file_list, target_coord):
    print("Extracting Analysis Cutouts...")
    n_pix = int(config.ANALYSIS_BOX_SIZE * config.SUPERSAMPLE_FACTOR)
    scale = config.PIXEL_SCALE / config.SUPERSAMPLE_FACTOR / 3600.0
    w_small = WCS(naxis=2)
    w_small.wcs.crpix = [n_pix/2, n_pix/2]
    w_small.wcs.crval = [target_coord.ra.deg, target_coord.dec.deg]
    w_small.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w_small.wcs.cdelt = [-scale, scale]
    w_small.wcs.pc = np.eye(2)
    cutouts = []
    for f in file_list:
        raw = load_data(f)
        d_jy = raw['data'] * config.MJY_SR_TO_JY; s_jy = raw['sigma'] * config.MJY_SR_TO_JY
        
        # Use aligned WCS if available, else raw
        wcs_use = f.get('corrected_wcs', raw['wcs'])
        
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            d_cut, _ = reproject_interp((d_jy, wcs_use), w_small, shape_out=(n_pix, n_pix))
            s_cut, _ = reproject_interp((s_jy, wcs_use), w_small, shape_out=(n_pix, n_pix))
        
        d_cut = np.nan_to_num(d_cut)
        s_cut = np.nan_to_num(s_cut, nan=np.inf)
        s_cut[(d_cut==0) & (s_cut==0)] = np.inf
        
        # FIX: Save raw_wcs (specifically the aligned one if possible) for PRF generation
        cutouts.append({
            'data': d_cut,
            'sigma': s_cut,
            'wcs': w_small,
            'raw_wcs': wcs_use,  # <--- CRITICAL: Saved for Solver
            'mjd': raw['mjd'],
            'filename': raw['filename'],
            'epoch_id': f.get('epoch_id', 0),
            'is_template': f.get('is_template', False)
        })
    return cutouts, w_small
