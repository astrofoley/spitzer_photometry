"""src/config.py"""
import os
import numpy as np

# --- Pipeline Control ---
CHANNEL = 'ch2'

# --- Target ---
TRANSIENT_RA = 197.45037
TRANSIENT_DEC = -23.38148

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data', 'raw')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
PRF_DIR = os.path.join(BASE_DIR, 'data', 'prfs')
DIAGNOSTIC_DIR = os.path.join(OUTPUT_DIR, 'diagnostics')
SOURCE_CATALOG_PATH = os.path.join(OUTPUT_DIR, 'source_catalog.fits')

# --- Analysis Geometry ---
PIXEL_SCALE = 1.22
STAMP_SIZE_PX = 20
BORDER_PX = 10
ANALYSIS_BOX_SIZE = STAMP_SIZE_PX + 2 * BORDER_PX
PRF_OVERSAMPLE = 100 # Standard IRAC PRF oversampling

# Super Resolution
SUPERSAMPLE_FACTOR = 2

# Buffer
SCENE_BUFFER_PX = 5

SPLIT_DATE_MJD = 58750.0
EPOCH_WINDOW_DAYS = 2.0

# --- Flux Conversion (Robust) ---
# Calculate pixel area in steradians manually to avoid UnitErrors
pixel_scale_deg = (PIXEL_SCALE / SUPERSAMPLE_FACTOR) / 3600.0
pixel_area_sr = (pixel_scale_deg * (np.pi / 180.0))**2
# MJy/sr -> Jy/pixel:  (1e6 Jy/sr) * (sr/pixel) = Jy/pixel
MJY_SR_TO_JY = 1e6 * pixel_area_sr

# --- Solver ---
INIT_LENGTH_SCALE = 2.0
INIT_VARIANCE = 1.0
GAIN = 3.7
READ_NOISE = 15.0
MAX_SCENE_PIXELS = 10000
