# Spitzer Photometry Pipeline

An automated pipeline for performing transient photometry on Spitzer Space Telescope images using Point Response Function (PRF) modeling and Generalized Least Squares (GLS) scene modeling.

## Features

* **Preprocessing:** Robust coordinate handling, astrometric alignment to deep templates, artifact masking, and cosmic ray rejection.
* **Scene Modeling:** Constructs a "super-resolution" static scene model from all available epochs using Generalized Least Squares (GLS).
* **Photometry:** Forward-modeling approach that projects high-resolution (100x oversampled) PRF models onto the scene grid, explicitly handling sub-pixel shifts, detector rotation, and flux conservation via WCS reprojection.
* **Diagnostics:** Generates multi-page PDF reports for residuals, epoch stacks, and lightcurves.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/yourusername/spitzer_photometry.git](https://github.com/yourusername/spitzer_photometry.git)
    cd spitzer_photometry
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Configuration

1.  Open `src/config.py`.
2.  Set `DATA_DIR` to the folder containing your Spitzer BCD images.
3.  Set `PRF_DIR` to the folder containing your PRF FITS files.
4.  Set `OUTPUT_DIR` and `DIAGNOSTIC_DIR` for results.

## Usage

Run the pipeline from the root directory:

```bash
