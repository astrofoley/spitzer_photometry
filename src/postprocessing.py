"""src/postprocessing.py"""
import os
import numpy as np
import pandas as pd
from astropy.io import fits
from . import config

def group_and_average(science_files, results):
    fluxes = results['transient_fluxes']
    errs = results['transient_errs']
    data = []
    for i, f in enumerate(science_files):
        mjd = f.get('mjd', 0.0)
        chan = f.get('channel', 1)
        data.append({'mjd': mjd, 'flux': fluxes[i], 'err': errs[i], 'channel': chan, 'filename': f['filename'], 'bcd_index': i})
    df = pd.DataFrame(data)
    df = df.sort_values('mjd').reset_index(drop=True)
    if not df.empty:
        print(f"[DEBUG] Post-processing {len(df)} records. MJD Range: {df['mjd'].min()} to {df['mjd'].max()}")
    epoch_ids = np.zeros(len(df), dtype=int)
    current_epoch = 1
    if len(df) > 0:
        last_mjd = df.iloc[0]['mjd']
        epoch_ids[0] = current_epoch
        for i in range(1, len(df)):
            mjd = df.iloc[i]['mjd']
            if (mjd - last_mjd) > config.EPOCH_WINDOW_DAYS: current_epoch += 1
            epoch_ids[i] = current_epoch
            last_mjd = mjd
    df['epoch_id'] = epoch_ids
    df.to_csv(os.path.join(config.DIAGNOSTIC_DIR, 'step4_epoch_grouping.csv'), index=False)
    summary = []
    if not df.empty:
        for (epoch, chan), group in df.groupby(['epoch_id', 'channel']):
            safe_errs = group['err'].copy()
            safe_errs[safe_errs == 0] = np.inf
            w = 1.0 / (safe_errs**2)
            w_sum = np.sum(w)
            if w_sum > 0:
                avg_flux = np.sum(group['flux'] * w) / w_sum
                avg_err = 1.0 / np.sqrt(w_sum)
                avg_mjd = np.average(group['mjd'], weights=w)
            else: avg_flux = 0.0; avg_err = 0.0; avg_mjd = group['mjd'].mean()
            summary.append({'epoch_id': epoch, 'channel': chan, 'mjd': avg_mjd, 'flux': avg_flux, 'err': avg_err, 'n_bcds': len(group)})
    df_summary = pd.DataFrame(summary)
    if not df_summary.empty: df_summary = df_summary.sort_values(['channel', 'mjd'])
    return df, df_summary

def save_outputs(df_bcds, df_summary, model_scene, output_dir):
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    df_bcds.to_csv(os.path.join(output_dir, 'flux_per_bcd.csv'), index=False)
    df_summary.to_csv(os.path.join(output_dir, 'flux_per_epoch.csv'), index=False)
    hdu = fits.PrimaryHDU(np.nan_to_num(model_scene))
    hdu.header['UNITS'] = 'Jy/pixel'
    hdu.writeto(os.path.join(output_dir, 'template_model_scene.fits'), overwrite=True)
    print(f"Saved results to {output_dir}")
