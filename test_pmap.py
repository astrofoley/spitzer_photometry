"""test_pmap.py — manual script; see tests/test_pmap_paths.py for pytest."""
import os
import numpy as np
from src import config
from src.pmap_correction import iracpc_pmap_corr

def test_correction():
    print("=== Testing P-Map Correction (IDL Replica) ===")
    
    pmap_dir = config.PMAP_DIR
    if not os.path.exists(pmap_dir):
        print(f"ERROR: {pmap_dir} not found.")
        return

    # TEST POINT: Must be within the Sweet Spot loaded in the previous logs
    # Log said: X-Range: [14.5010, 15.4990]
    flux_obs = 100.0
    x_test = 15.0  # Center of sweet spot
    y_test = 15.0
    channel = 'ch1'

    print(f"\nInput Flux: {flux_obs}")
    print(f"Position:   ({x_test}, {y_test})")
    print(f"Channel:    {channel}")

    # Run Correction
    try:
        corr_flux = iracpc_pmap_corr(flux_obs, x_test, y_test, channel, pmap_dir=pmap_dir, threshold_occ=False)
        
        print(f"\n--- Result ---")
        print(f"Corrected Flux:    {corr_flux:.4f}")
        
        if np.isnan(corr_flux):
            print("RESULT: NaN (Coordinate is outside the map sweet spot)")
        else:
            gain = flux_obs / corr_flux
            print(f"Implied Gain:      {gain:.4f}")

    except Exception as e:
        print(f"\nFAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_correction()
