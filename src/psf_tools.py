"""
src/psf_tools.py

Handles loading of high-resolution (100x oversampled) Spitzer IRAC PRFs,
selection of the appropriate PRF based on detector position, and generation
of design matrix columns via sub-pixel shifting and binning.
"""

import os
import glob
import re
import numpy as np
from astropy.io import fits
from scipy.ndimage import shift

def load_100x_prf(filepath):
    """
    Loads a high-res (1/100th pixel) PRF FITS file.
    Normalizes the PRF so the total sum is 1.0 (flux conservation).
    
    Parameters:
    -----------
    filepath : str
        Path to the FITS file (e.g., '.../apex_sh_IRACPC1_col025_row025_x100.fits')

    Returns:
    --------
    data : np.ndarray
        The normalized 2D PRF array.
    """
    if not filepath or not os.path.exists(filepath):
        raise FileNotFoundError(f"PRF file not found: {filepath}")
        
    with fits.open(filepath) as hdul:
        # IRAC PRF files usually store data in the primary HDU (ext 0)
        data = hdul[0].data.astype(np.float64)
        
        # Normalize flux
        total_flux = np.sum(data)
        if total_flux > 0:
            data /= total_flux
            
    return data

def get_nearest_prf_file(x_det, y_det, channel_number, prf_dir):
    """
    Finds the PRF file in the grid closest to the given detector (x,y).
    
    Parameters:
    -----------
    x_det, y_det : float
        The pixel coordinates of the source on the science BCD (0-255).
    channel_number : int
        1 (3.6um) or 2 (4.5um).
    prf_dir : str
        Path to the directory containing the .fits files.
        
    Returns:
    --------
    str : Full path to the best matching PRF file.
    """
    # Expected pattern: apex_sh_IRACPC1_col025_row025_x100.fits
    # Note: 'PC1' corresponds to Channel 1, 'PC2' to Channel 2.
    glob_pattern = os.path.join(prf_dir, f"*IRACPC{channel_number}*x100.fits")
    candidates = glob.glob(glob_pattern)
    
    if not candidates:
        raise FileNotFoundError(f"No PRF files found for Channel {channel_number} in {prf_dir}")

    best_file = None
    min_dist_sq = float('inf')

    # Regex to extract row and col from filename
    # Matches 'col' followed by digits, and 'row' followed by digits
    pattern = re.compile(r"col(\d+)_row(\d+)")

    for f in candidates:
        fname = os.path.basename(f)
        match = pattern.search(fname)
        
        if match:
            # Extract grid center coordinates from filename
            grid_col = int(match.group(1)) # x coordinate
            grid_row = int(match.group(2)) # y coordinate
            
            # Calculate Euclidean distance squared
            dist_sq = (grid_col - x_det)**2 + (grid_row - y_det)**2
            
            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
                best_file = f
    
    if best_file is None:
         raise ValueError(f"Could not parse row/col from filenames in {prf_dir}")

    # Optional: Log the selection for debugging
    # print(f"Selected PRF: {os.path.basename(best_file)} for source at ({x_det:.1f}, {y_det:.1f})")
    
    return best_file

def generate_design_column(wcs_bcd, prf_100x, ra, dec, output_shape):
    """
    Generates the point source model for the Science Image (Design Matrix Column).
    
    1. Converts RA/Dec to Science Pixel coordinates (float).
    2. Shifts the 100x PRF to match the sub-pixel phase.
    3. Bins (block-sums) the 100x PRF down to the 1x Science resolution.
    
    Parameters:
    -----------
    wcs_bcd : astropy.wcs.WCS
        WCS of the science image.
    prf_100x : np.ndarray
        The loaded high-res PRF array.
    ra, dec : float
        Target coordinates in degrees.
    output_shape : tuple
        Shape of the science stamp (height, width).
        
    Returns:
    --------
    np.ndarray (1D)
        Flattened array of the point source model.
    """
    oversample = 100
    
    # 1. Get float pixel position in Science Frame
    x_sci, y_sci = wcs_bcd.world_to_pixel_values(ra, dec)
    
    # 2. Determine integer center and fractional offset
    # We round to the nearest integer pixel to determine the 'placement' of the PRF stamp
    x_int = int(np.round(x_sci))
    y_int = int(np.round(y_sci))
    
    # The sub-pixel offset (in science pixels)
    # If x_sci is 10.3, x_int is 10, dx is +0.3 (Shift Right)
    dx = x_sci - x_int
    dy = y_sci - y_int
    
    # Convert offset to High-Res pixels
    shift_x = dx * oversample
    shift_y = dy * oversample
    
    # 3. Shift the 100x PRF
    # This aligns the PRF peak exactly with the source center
    # order=1 (linear) is usually sufficient for oversampled data and faster/safer than spline
    shifted_prf = shift(prf_100x, (shift_y, shift_x), order=1, mode='constant', cval=0.0)
    
    # 4. Bin (Downsample) to Science Resolution
    # We must ensure the array dimensions are multiples of 100 for reshaping.
    h_high, w_high = shifted_prf.shape
    
    # Crop to nearest multiple of 100 (centered)
    h_crop = (h_high // oversample) * oversample
    w_crop = (w_high // oversample) * oversample
    
    # Calculate crop indices
    y_start_crop = (h_high - h_crop) // 2
    x_start_crop = (w_high - w_crop) // 2
    
    cropped_prf = shifted_prf[y_start_crop : y_start_crop + h_crop,
                              x_start_crop : x_start_crop + w_crop]
    
    # Rebin logic: Reshape -> Sum -> Sum
    new_h = h_crop // oversample
    new_w = w_crop // oversample
    
    # Shape: (New_H, 100, New_W, 100)
    reshaped = cropped_prf.reshape(new_h, oversample, new_w, oversample)
    
    # Sum over the 100x100 blocks
    prf_model_small = reshaped.sum(axis=3).sum(axis=1)
    
    # 5. Place the small PRF stamp into the full Science Image
    full_image = np.zeros(output_shape)
    
    # Calculate placement coordinates
    # The prf_model_small is centered on (new_w//2, new_h//2)
    # We place that center at (x_int, y_int) in the full image
    
    start_y = y_int - (new_h // 2)
    start_x = x_int - (new_w // 2)
    
    # Slicing logic for boundaries (handle if PRF goes off edge of stamp)
    h_out, w_out = output_shape
    
    y_img_start = max(0, start_y)
    y_img_end = min(h_out, start_y + new_h)
    x_img_start = max(0, start_x)
    x_img_end = min(w_out, start_x + new_w)
    
    y_prf_start = y_img_start - start_y
    y_prf_end = y_prf_start + (y_img_end - y_img_start)
    x_prf_start = x_img_start - start_x
    x_prf_end = x_prf_start + (x_img_end - x_img_start)
    
    # Only assign if there is overlap
    if y_img_end > y_img_start and x_img_end > x_img_start:
        full_image[y_img_start:y_img_end, x_img_start:x_img_end] = \
            prf_model_small[y_prf_start:y_prf_end, x_prf_start:x_prf_end]
            
    return full_image.flatten()
