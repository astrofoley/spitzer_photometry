#!/usr/bin/env python3
"""
Standalone template dipole χ² scan (shift fitted host+nucleus+BG model along dipole).

Runs the same fit stage as main.py (without full diagnostic PDFs), then:
  - fit once, build host+nucleus+BG submodel per template BCD
  - coarse shifts 0 … 3 px along dipole, step 0.1
  - polynomial fit to χ²(s); fine scan ±0.2 px around vertex, step 0.01
  - writes JSON + χ² PNG + residual-gallery PNG under DIAGNOSTIC_DIR

Usage (from repo root):
  python scripts/template_dipole_chi2_scan.py
  python scripts/template_dipole_chi2_scan.py --skip-pre-analysis

Does not require nonzero host_core_flux; it shifts the fitted structured
submodel (host+nucleus+BG = GP+host+BG in current implementation).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src import config  # noqa: E402
from src import dipole_chi2_scan  # noqa: E402
from src import pipeline_fit  # noqa: E402


def main():
    p = argparse.ArgumentParser(description='Template BCD dipole χ² scan')
    p.add_argument(
        '--skip-pre-analysis',
        action='store_true',
        help='Skip PRE_ANALYSIS_CHECK.png (faster)',
    )
    p.add_argument(
        '--json-out',
        default=None,
        help='JSON path (default: DIAGNOSTIC_DIR/template_dipole_chi2_scan.json)',
    )
    p.add_argument(
        '--plot-out',
        default=None,
        help='PNG path (default: DIAGNOSTIC_DIR/template_dipole_chi2_scan.png)',
    )
    p.add_argument(
        '--resid-out',
        default=None,
        help='Residual gallery PNG path (default: DIAGNOSTIC_DIR/template_dipole_shifted_residuals.png)',
    )
    args = p.parse_args()

    os.makedirs(config.DIAGNOSTIC_DIR, exist_ok=True)
    fit = pipeline_fit.run_pipeline_fit_core(skip_pre_analysis_check=args.skip_pre_analysis)
    if fit is None:
        print('Fit stage failed.', file=sys.stderr)
        sys.exit(1)

    scan = dipole_chi2_scan.compute_dipole_chi2_scan(
        fit['cutouts'],
        fit['results'],
        fit['all_stars'],
        fit['full_flux_list'],
        fit['stretch_mask'],
        coarse_step=0.1,
        coarse_max=3.0,
        fine_half_width=0.2,
        fine_step=0.01,
        poly_degree=2,
    )

    jpath = args.json_out or os.path.join(
        config.DIAGNOSTIC_DIR, 'template_dipole_chi2_scan.json',
    )
    with open(jpath, 'w', encoding='utf-8') as f:
        json.dump(dipole_chi2_scan.json_sanitize(scan), f, indent=2)
    print(f'Wrote {jpath}')

    ppath = args.plot_out or os.path.join(
        config.DIAGNOSTIC_DIR, 'template_dipole_chi2_scan.png',
    )
    dipole_chi2_scan.plot_chi2_scan(scan, ppath, title='Template dipole χ² scan (standalone)')
    print(f'Wrote {ppath}')
    rpath = args.resid_out or os.path.join(
        config.DIAGNOSTIC_DIR, 'template_dipole_shifted_residuals.png',
    )
    dipole_chi2_scan.plot_shifted_residual_gallery(
        scan,
        fit['cutouts'],
        fit['results'],
        fit['all_stars'],
        fit['full_flux_list'],
        fit['stretch_mask'],
        rpath,
    )
    print(f'Wrote {rpath}')

    sec = scan.get('dipole_chi2_refinement', {})
    if sec.get('skipped'):
        print(f"Scan skipped: {sec.get('reason')}")
        sys.exit(2)


if __name__ == '__main__':
    main()
