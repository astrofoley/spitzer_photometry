#!/usr/bin/env python3
"""
Optional A/B hooks for residual QA: supersample factor, PRF directory, tiny WCS shifts.

Run from repo root, e.g.:
  python scripts/ab_residual_experiments.py --print-dipole-synthetic

This script does not execute the full pipeline; it documents knobs and runs a tiny
synthetic check for dipole metrics when numpy is available.
"""
from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def main():
    p = argparse.ArgumentParser(description='Residual A/B experiment helpers')
    p.add_argument(
        '--print-dipole-synthetic',
        action='store_true',
        help='Run dipole_moment_xy on a shifted Gaussian blob (sanity check).',
    )
    args = p.parse_args()

    if args.print_dipole_synthetic:
        import numpy as np
        from src import residual_metrics

        h, w = 32, 32
        yy, xx = np.mgrid[0:h, 0:w]
        z = np.exp(-((xx - 10.0) ** 2 + (yy - 16.0) ** 2) / (2 * 2.5 ** 2))
        m = np.ones_like(z, dtype=bool)
        dx, dy, dm = residual_metrics.dipole_moment_xy(z, m)
        print(f'dipole_moment_xy: mx={dx:.4f} my={dy:.4f} |m|={dm:.4f}')
        return

    p.print_help()


if __name__ == '__main__':
    main()
