"""src/diagnostics.py"""
import os
import json
import math
import warnings
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import AsinhNorm
from reproject import reproject_interp
from . import config
from . import solver
from . import residual_metrics


def get_robust_limits(data, pmin=1, pmax=99, mask=None):
    if mask is None:
        valid = data[np.isfinite(data) & (data != 0)]
    else:
        valid = data[mask & np.isfinite(data) & (data != 0)]
    if len(valid) == 0:
        return 0, 1
    lo, hi = np.percentile(valid, [pmin, pmax])
    if hi == lo:
        hi += 1e-6
    return lo, hi


def get_percentile_limits(data, mask=None, lo=None, hi=None):
    """vmin/vmax from percentiles of finite values (default 1–95 from config)."""
    if lo is None:
        lo = float(getattr(config, 'DIAGNOSTIC_IMSHOW_PERCENTILES_LO', 1.0))
    if hi is None:
        hi = float(getattr(config, 'DIAGNOSTIC_IMSHOW_PERCENTILES_HI', 95.0))
    d = np.asarray(data, dtype=float)
    if mask is None:
        v = d[np.isfinite(d)]
    else:
        v = d[mask & np.isfinite(d)]
    if len(v) < 4:
        return 0.0, 1.0
    vmin, vmax = np.percentile(v, [lo, hi])
    if vmax == vmin:
        vmax = vmin + 1e-12
    return float(vmin), float(vmax)


def _imshow_linear_percentile(ax, data, mask=None, cmap='RdBu_r', **kwargs):
    vmin, vmax = get_percentile_limits(data, mask=mask)
    return ax.imshow(
        np.asarray(data, dtype=float),
        origin='lower',
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        **kwargs,
    )


def galaxy_disk_annulus_mask(scene_wcs, scene_shape):
    """
    Annulus in analysis pixels around GALAXY_EXTENDED_CENTER_* when set;
    otherwise around TRANSIENT_RA/DEC (QA disk region).
    """
    ra = getattr(config, 'GALAXY_EXTENDED_CENTER_RA', None)
    dec = getattr(config, 'GALAXY_EXTENDED_CENTER_DEC', None)
    h, w = int(scene_shape[0]), int(scene_shape[1])
    yy, xx = np.mgrid[0:h, 0:w].astype(float)
    if scene_wcs is None:
        return np.ones((h, w), dtype=bool)
    if ra is None or dec is None:
        ra, dec = float(config.TRANSIENT_RA), float(config.TRANSIENT_DEC)
    else:
        ra, dec = float(ra), float(dec)
    ox, oy = scene_wcs.world_to_pixel_values(ra, dec)
    ri = float(getattr(config, 'GALAXY_QA_ANNULUS_INNER_PX', 4.0))
    ro = float(getattr(config, 'GALAXY_QA_ANNULUS_OUTER_PX', 14.0))
    r = np.hypot(xx - ox, yy - oy)
    return (r >= ri) & (r <= ro)


def _asinh_linear_width(data, mask=None, frac=None, width_pmin=2, width_pmax=98):
    """AsinhNorm linear_width from robust span of positive/negative flux."""
    if frac is None:
        frac = float(getattr(config, 'DIAGNOSTIC_ASINH_WIDTH_FRAC', 0.12))
    lo, hi = get_robust_limits(data, width_pmin, width_pmax, mask=mask)
    span = max(float(hi - lo), 1e-30)
    return max(frac * span, max(abs(lo), abs(hi)) * 0.02 + 1e-20)


def _imshow_flux_asinh(
    ax, data, mask=None, cmap='gray', interpolation='nearest',
    *, pmin=1, pmax=99, width_pmin=2, width_pmax=98, width_frac=None,
):
    d = np.asarray(data, dtype=float)
    lw = _asinh_linear_width(d, mask=mask, frac=width_frac, width_pmin=width_pmin, width_pmax=width_pmax)
    lo, hi = get_robust_limits(d, pmin, pmax, mask=mask)
    norm = AsinhNorm(linear_width=lw, vmin=lo, vmax=hi)
    return ax.imshow(d, origin='lower', cmap=cmap, norm=norm, interpolation=interpolation)


def _bcd_flux_percentiles():
    t = getattr(config, 'DIAGNOSTIC_BCD_ROBUST_PERCENTILES', (0.5, 99.5))
    if t is None or len(t) != 2:
        return 0.5, 99.5
    return float(t[0]), float(t[1])


def _bcd_asinh_width_frac():
    w = getattr(config, 'DIAGNOSTIC_BCD_ASINH_WIDTH_FRAC', None)
    if w is None:
        return None
    return float(w)


def _imshow_bcd_flux_asinh(ax, data, mask=None, cmap='gray', interpolation='nearest'):
    p_lo, p_hi = _bcd_flux_percentiles()
    wf = _bcd_asinh_width_frac()
    return _imshow_flux_asinh(
        ax, data, mask=mask, cmap=cmap, interpolation=interpolation,
        pmin=p_lo, pmax=p_hi, width_pmin=p_lo, width_pmax=p_hi, width_frac=wf,
    )


def _residual_symmetric_limits(resid, mask=None):
    """Symmetric vmin/vmax from configured residual percentiles (default p1/p99)."""
    r = np.asarray(resid, dtype=float)
    if mask is not None:
        v = r[mask & np.isfinite(r)]
    else:
        v = r[np.isfinite(r)]
    if len(v) == 0:
        return -1.0, 1.0
    plo = float(getattr(config, 'DIAGNOSTIC_RESIDUAL_PERCENTILES_LO', 1.0))
    phi = float(getattr(config, 'DIAGNOSTIC_RESIDUAL_PERCENTILES_HI', 99.0))
    lo, hi = np.percentile(v, [plo, phi])
    lim = max(abs(float(lo)), abs(float(hi)), 1e-15)
    return -lim, lim


def _sigma_residual_map(resid, sigma, valid_mask):
    with np.errstate(divide='ignore', invalid='ignore'):
        z = np.asarray(resid, dtype=float) / np.clip(np.asarray(sigma, dtype=float), 1e-30, None)
    z = np.where(valid_mask, z, np.nan)
    return z


def _sigma_display_limits(z, valid_mask, cap=None):
    if cap is None:
        cap = float(getattr(config, 'DIAGNOSTIC_RESID_SIGMA_DISPLAY_CAP', 6.0))
    v = z[valid_mask & np.isfinite(z)]
    if len(v) == 0:
        return 3.0
    lim = float(np.nanpercentile(np.abs(v), 98))
    return float(np.clip(max(lim, 0.5), 0.5, cap))


def diagnostic_stretch_mask(cutouts, stamp_shape):
    """Pixels used for robust vmin/vmax (exclude CR/sigma-masked and zero data)."""
    h, w = int(stamp_shape[0]), int(stamp_shape[1])
    m = np.ones((h, w), dtype=bool)
    for c in cutouts:
        d = np.asarray(c['data'])
        sig = np.asarray(c['sigma'])
        if d.shape != (h, w):
            continue
        m &= (d != 0) & np.isfinite(sig) & (sig < 1e20)
    return m


def _lightcurve_row_is_placeholder(flux, err, is_template):
    """Template epochs pinned to zero flux — hide from plots when errors are dummy."""
    if not is_template:
        return False
    if abs(float(flux)) > 1e-18:
        return False
    e = float(err)
    return (e <= 1e-18) or (abs(e - 1.0) < 0.05)

def plot_pre_analysis_check(stack, wcs, stars, fluxes, target_coord, stretch_mask=None):
    """
    Plots ONLY the deep template with detected stars marked.
    Uses asinh stretch by default for faint structure near bright cores.
    """
    fig, ax = plt.subplots(figsize=(12, 12))
    _imshow_flux_asinh(ax, stack, mask=stretch_mask)
    ax.set_title("Deep Template - Source Check (asinh stretch)")

    tx, ty = wcs.world_to_pixel(target_coord)
    ax.plot(tx, ty, 'rx', markersize=15, markeredgewidth=3, label='Target')

    for i, (star, flux) in enumerate(zip(stars, fluxes)):
        sx, sy = wcs.world_to_pixel(star)
        if flux > 1e-4:
            color = 'cyan'
            marker = 'o'
            size = 12
        else:
            color = 'orange'
            marker = 'x'
            size = 8
        ax.plot(sx, sy, marker=marker, color=color, markersize=size, fillstyle='none', markeredgewidth=1.5)
        if flux > 1e-4:
            ax.text(sx + 4, sy + 4, f"{i}", color=color, fontsize=10, fontweight='bold')

    ax.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join(config.DIAGNOSTIC_DIR, 'PRE_ANALYSIS_CHECK.png'), dpi=150)
    plt.close()

def _transient_only_stamp(results, cutouts, frame_index):
    """PRF × f_epoch on the analysis grid (no background)."""
    c = cutouts[frame_index]
    if c.get('is_template'):
        return None
    p = solver.predict_cutout_model(
        results, cutouts, [], [], frame_index,
        include_gp=False, include_stars=False, include_transient=True,
        include_host=False,
    )
    bg = float(results['epoch_backgrounds'][int(c['epoch_id'])])
    return np.asarray(p, dtype=float) - bg


def plot_fit_template_with_stars(
    template,
    wcs,
    stars,
    fluxes,
    target_coord,
    stretch_mask=None,
    results=None,
    cutouts=None,
    transient_overlay_frame_index=None,
):
    """
    Left: reconstructed scene (GP + mean star PRFs; transient not in reconstruct).
    Right: transient-only stamp (PRF × f_epoch) on its own scale when available — no alpha blend.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    ax0, ax1 = axes[0], axes[1]

    _imshow_flux_asinh(ax0, template, mask=stretch_mask, cmap='gray', pmin=1, pmax=95, width_pmin=1, width_pmax=95)
    tx, ty = wcs.world_to_pixel(target_coord)
    ax0.plot(tx, ty, 'rx', markersize=15, markeredgewidth=3, label='Transient Loc')
    for i, (star, flux) in enumerate(zip(stars, fluxes)):
        sx, sy = wcs.world_to_pixel(star)
        color = 'cyan'
        ms = 10 + np.log10(max(flux, 1e-9)) * 2
        ms = max(5, min(ms, 20))
        ax0.plot(sx, sy, 'o', color=color, markersize=ms, fillstyle='none', markeredgewidth=2)
        ax0.text(sx + 3, sy + 3, f"{i}", color=color, fontsize=12, fontweight='bold')
    ax0.set_title("Reconstructed template (GP + stars; transient not in reconstruct)")
    ax0.legend(loc='upper right')

    tmap = None
    if (
        results is not None
        and cutouts is not None
        and transient_overlay_frame_index is not None
    ):
        tmap = _transient_only_stamp(results, cutouts, int(transient_overlay_frame_index))

    if tmap is not None and tmap.shape == template.shape:
        tm = stretch_mask if stretch_mask is not None else np.isfinite(tmap)
        tclip = np.clip(tmap, 0.0, None)
        _imshow_flux_asinh(ax1, tclip, mask=tm, cmap='Oranges', pmin=1, pmax=95, width_pmin=1, width_pmax=95)
        ax1.plot(tx, ty, 'rx', markersize=12, markeredgewidth=2)
        ax1.set_title(f"Transient only PRF×f (frame {transient_overlay_frame_index})")
    else:
        ax1.text(0.5, 0.5, "No transient stamp\n(template epoch or missing data)", ha='center', va='center', transform=ax1.transAxes)
        ax1.set_axis_off()

    for ax in axes:
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(config.DIAGNOSTIC_DIR, 'DIAGNOSTIC_FIT_TEMPLATE.png'), dpi=150)
    plt.close()

def _bcd_valid_mask(c, stretch_mask):
    d = np.asarray(c['data'])
    sig = np.asarray(c['sigma'])
    m = (d != 0) & np.isfinite(sig) & (sig < 1e20)
    if stretch_mask is not None and stretch_mask.shape == d.shape:
        m &= stretch_mask
    return m


def plot_bcd_residuals(cutouts, results, stars, star_fluxes, deep_stack, stretch_mask=None):
    """
    Per-BCD: row0 = flux (asinh) + residuals (Jy, robust symmetric).
    row1 = same residuals in σ = (data−model)/sigma units (shared color scale per frame).
    """
    print("   [Diagnostics] Generating Detailed BCD Residuals...")

    sci_indices = [i for i, c in enumerate(cutouts) if not c.get('is_template', False)]
    tpl_indices = [i for i, c in enumerate(cutouts) if c.get('is_template', False)]

    rng = np.random.default_rng(42)
    sel_sci = rng.choice(sci_indices, size=min(5, len(sci_indices)), replace=False)
    sel_tpl = rng.choice(tpl_indices, size=min(5, len(tpl_indices)), replace=False)
    selection = np.sort(np.concatenate([sel_sci, sel_tpl]))

    gp_scene = np.asarray(results.get('gp_scene', results['model_scene']))
    epoch_bgs = results['epoch_backgrounds']
    transient_fluxes = results['transient_fluxes']

    pdf_path = os.path.join(config.DIAGNOSTIC_DIR, 'DIAGNOSTIC_DETAILED_RESIDUALS.pdf')

    with PdfPages(pdf_path) as pdf:
        for k in selection:
            c = cutouts[k]
            data = np.asarray(c['data'], dtype=float)
            sig = np.asarray(c['sigma'], dtype=float)
            vm = _bcd_valid_mask(c, stretch_mask)

            mod_full = solver.predict_cutout_model(
                results, cutouts, stars, star_fluxes, k,
                include_transient=True, include_stars=True,
            )
            mod_no_trans = solver.predict_cutout_model(
                results, cutouts, stars, star_fluxes, k,
                include_transient=False, include_stars=True,
            )
            mod_gp_only = solver.predict_cutout_model(
                results, cutouts, stars, star_fluxes, k,
                include_transient=False, include_stars=False, include_host=False,
            )

            t_flux = transient_fluxes[k]

            res_deep = data - deep_stack
            res_no_trans = data - mod_no_trans
            res_w_trans = data - mod_full
            mod_gp_host = solver.predict_cutout_model(
                results, cutouts, stars, star_fluxes, k,
                include_transient=False, include_stars=False, include_host=True,
            )
            res_stars = data - mod_gp_host

            fig = plt.figure(figsize=(45, 10))
            gs = fig.add_gridspec(2, 9, height_ratios=[1.0, 0.9], hspace=0.22, wspace=0.12)
            axes0 = [fig.add_subplot(gs[0, j]) for j in range(9)]
            ax_s1 = fig.add_subplot(gs[1, 0:3])
            ax_s2 = fig.add_subplot(gs[1, 3:6])
            ax_s3 = fig.add_subplot(gs[1, 6:9])

            _imshow_bcd_flux_asinh(axes0[0], data, mask=vm)
            axes0[0].set_title(f"BCD (asinh)\n{c['filename'][-15:]}")

            _imshow_bcd_flux_asinh(axes0[1], deep_stack, mask=stretch_mask)
            axes0[1].set_title("Deep stack (asinh)")

            rv_d = _residual_symmetric_limits(res_deep, mask=vm)
            axes0[2].imshow(res_deep, origin='lower', vmin=rv_d[0], vmax=rv_d[1], cmap='RdBu_r')
            axes0[2].set_title("Resid vs deep (Jy)")

            _imshow_bcd_flux_asinh(axes0[3], mod_no_trans, mask=vm)
            axes0[3].set_title("Model no trans (asinh)")

            rv_nt = _residual_symmetric_limits(res_no_trans, mask=vm)
            axes0[4].imshow(res_no_trans, origin='lower', vmin=rv_nt[0], vmax=rv_nt[1], cmap='RdBu_r')
            axes0[4].set_title("Resid no trans (Jy)")

            _imshow_bcd_flux_asinh(axes0[5], mod_full, mask=vm)
            axes0[5].set_title(f"Model full (asinh)\nF={t_flux:.1e}")

            rv_f = _residual_symmetric_limits(res_w_trans, mask=vm)
            axes0[6].imshow(res_w_trans, origin='lower', vmin=rv_f[0], vmax=rv_f[1], cmap='RdBu_r')
            axes0[6].set_title("Resid full (Jy)")

            _imshow_bcd_flux_asinh(axes0[7], mod_gp_only, mask=vm)
            axes0[7].set_title("GP + BG (asinh)")

            rv_g = _residual_symmetric_limits(res_stars, mask=vm)
            axes0[8].imshow(res_stars, origin='lower', vmin=rv_g[0], vmax=rv_g[1], cmap='RdBu_r')
            axes0[8].set_title("Resid vs GP+BG+host (Jy)")

            z_nt = _sigma_residual_map(res_no_trans, sig, vm)
            z_f = _sigma_residual_map(res_w_trans, sig, vm)
            z_g = _sigma_residual_map(res_stars, sig, vm)
            lim = max(
                _sigma_display_limits(z_nt, vm),
                _sigma_display_limits(z_f, vm),
                _sigma_display_limits(z_g, vm),
            )

            ax_s1.imshow(z_nt, origin='lower', vmin=-lim, vmax=lim, cmap='RdBu_r')
            ax_s1.set_title(f"Resid no trans (σ); |z|≤{lim:.1f}σ")
            ax_s2.imshow(z_f, origin='lower', vmin=-lim, vmax=lim, cmap='RdBu_r')
            ax_s2.set_title(f"Resid full (σ); |z|≤{lim:.1f}σ")
            ax_s3.imshow(z_g, origin='lower', vmin=-lim, vmax=lim, cmap='RdBu_r')
            ax_s3.set_title(f"Resid vs GP+BG+host (σ); |z|≤{lim:.1f}σ")

            for ax in axes0 + [ax_s1, ax_s2, ax_s3]:
                ax.axis('off')
            pdf.savefig(fig)
            plt.close()

def plot_epoch_stacks(cutouts, results, target_coord, stars, star_fluxes, stretch_mask=None):
    """
    Per epoch: page A = median(data − full pred) in Jy (percentile-scaled RdBu).
    Page B = median per-pixel (resid/sigma) across BCDs (σ units).
    """
    print("   [Diagnostics] Generating Epoch Stacks PDF...")
    pdf_path = os.path.join(config.DIAGNOSTIC_DIR, 'DIAGNOSTIC_EPOCH_STACKS.pdf')

    epoch_ids = np.array([c['epoch_id'] for c in cutouts])
    unique_epochs = np.unique(epoch_ids)
    stack_wcs, stack_shape = _stack_grid(results, cutouts)

    with PdfPages(pdf_path) as pdf:
        for ep in unique_epochs:
            mask = (epoch_ids == ep)
            subset_indices = np.where(mask)[0]

            stack_resid = []
            stack_z = []
            for idx in subset_indices:
                c = cutouts[idx]
                pred = solver.predict_cutout_model(
                    results, cutouts, stars, star_fluxes, idx,
                    include_transient=True, include_stars=True,
                )
                res = np.asarray(c['data'], dtype=float) - pred
                stack_resid.append(_to_stack_grid(res, c['wcs'], stack_wcs, stack_shape))
                vm = _bcd_valid_mask(c, stretch_mask)
                z = _sigma_residual_map(res, np.asarray(c['sigma'], dtype=float), vm)
                stack_z.append(_to_stack_grid(z, c['wcs'], stack_wcs, stack_shape))

            if not stack_resid:
                continue

            cube_r = np.stack(stack_resid, axis=0)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                med_stack = np.nanmedian(cube_r, axis=0)
            cube_z = np.stack(stack_z, axis=0)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                med_z = np.nanmedian(cube_z, axis=0)

            sm = stretch_mask if stretch_mask is not None else np.isfinite(med_stack)
            rv0, rv1 = _residual_symmetric_limits(med_stack, mask=sm)
            fig1, ax1 = plt.subplots(figsize=(8, 8))
            im1 = ax1.imshow(med_stack, origin='lower', vmin=rv0, vmax=rv1, cmap='RdBu_r')
            plt.colorbar(im1, ax=ax1, label='Median resid (Jy)')
            lo_p = float(getattr(config, 'DIAGNOSTIC_IMSHOW_PERCENTILES_LO', 1.0))
            hi_p = float(getattr(config, 'DIAGNOSTIC_IMSHOW_PERCENTILES_HI', 95.0))
            ax1.set_title(
                f"Epoch {ep} median resid Jy (data−full pred; n={len(stack_resid)})\n"
                f"display percentiles p{lo_p:.0f}–p{hi_p:.0f}"
            )
            h, w = med_stack.shape
            ax1.plot(w / 2, h / 2, 'gx', markersize=12, markeredgewidth=2)
            ax1.axis('off')
            plt.tight_layout()
            pdf.savefig(fig1)
            plt.close()

            zm = sm & np.isfinite(med_z)
            zmin, zmax = _residual_symmetric_limits(med_z, mask=zm)
            fig2, ax2 = plt.subplots(figsize=(8, 8))
            im2 = ax2.imshow(med_z, origin='lower', vmin=zmin, vmax=zmax, cmap='RdBu_r')
            plt.colorbar(im2, ax=ax2, label='Median resid / σ')
            ax2.set_title(
                f"Epoch {ep} median (resid/σ); n={len(stack_resid)}; "
                f"p{lo_p:.0f}–p{hi_p:.0f} display"
            )
            ax2.plot(w / 2, h / 2, 'gx', markersize=12, markeredgewidth=2)
            ax2.axis('off')
            plt.tight_layout()
            pdf.savefig(fig2)
            plt.close()


def plot_stacked_residuals_with_without_transient(
    cutouts, results, stars, star_fluxes, stretch_mask=None
):
    """Median-stack residuals on the common scene grid, comparing transient on vs off in the predictor.

    For each frame, forms ``data - predict(...)`` with ``include_transient`` True vs False (stars+GP+host
    unchanged), reprojects to ``results['scene_wcs']``, masks invalid pixels, then takes pixelwise
    median across epochs. Writes ``STACKED_RESIDUALS_WITH_WITHOUT_TRANSIENT.pdf`` with a third
    panel showing the difference (no transient minus with transient), i.e. the stack-level
    imprint of the transient term.
    """
    print("   [Diagnostics] Generating stacked residuals with/without transient...")
    pdf_path = os.path.join(config.DIAGNOSTIC_DIR, 'STACKED_RESIDUALS_WITH_WITHOUT_TRANSIENT.pdf')
    stack_wcs, stack_shape = _stack_grid(results, cutouts)
    cube_with = []
    cube_without = []
    for i, c in enumerate(cutouts):
        data = np.asarray(c['data'], dtype=float)
        pred_with = solver.predict_cutout_model(
            results, cutouts, stars, star_fluxes, i,
            include_transient=True, include_stars=True, include_gp=True, include_host=True,
        )
        pred_without = solver.predict_cutout_model(
            results, cutouts, stars, star_fluxes, i,
            include_transient=False, include_stars=True, include_gp=True, include_host=True,
        )
        resid_with = data - pred_with
        resid_without = data - pred_without
        vm = _bcd_valid_mask(c, stretch_mask).astype(float)
        rw = _to_stack_grid(resid_with, c['wcs'], stack_wcs, stack_shape)
        rn = _to_stack_grid(resid_without, c['wcs'], stack_wcs, stack_shape)
        mm = _to_stack_grid(vm, c['wcs'], stack_wcs, stack_shape)
        use = np.asarray(mm, dtype=float) > 0.5
        cube_with.append(np.where(use, np.asarray(rw, dtype=float), np.nan))
        cube_without.append(np.where(use, np.asarray(rn, dtype=float), np.nan))
    med_with = np.nanmedian(np.stack(cube_with, axis=0), axis=0)
    med_without = np.nanmedian(np.stack(cube_without, axis=0), axis=0)
    delta = med_without - med_with
    with PdfPages(pdf_path) as pdf:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        v0, v1 = _residual_symmetric_limits(med_without)
        im0 = axes[0].imshow(med_without, origin='lower', cmap='RdBu_r', vmin=v0, vmax=v1)
        axes[0].set_title('Median residual (no transient)')
        axes[0].axis('off')
        plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
        v2, v3 = _residual_symmetric_limits(med_with)
        im1 = axes[1].imshow(med_with, origin='lower', cmap='RdBu_r', vmin=v2, vmax=v3)
        axes[1].set_title('Median residual (with transient)')
        axes[1].axis('off')
        plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
        v4, v5 = _residual_symmetric_limits(delta)
        im2 = axes[2].imshow(delta, origin='lower', cmap='RdBu_r', vmin=v4, vmax=v5)
        axes[2].set_title('Difference: no transient - with transient')
        axes[2].axis('off')
        plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)
    print(f"   [Diagnostics] Wrote {pdf_path}")


def _template_cutout_indices(cutouts):
    return [i for i, c in enumerate(cutouts) if c.get('is_template')]


def _stack_grid(results, cutouts):
    """Common North-up stack grid for cross-frame medians in native-fit mode."""
    scene_wcs = results.get('scene_wcs')
    scene_shape = results.get('scene_shape')
    if scene_wcs is None or scene_shape is None:
        return None, None
    if bool(getattr(config, 'FIT_ON_NATIVE_PIXELS', False)):
        return scene_wcs, tuple(scene_shape)
    if cutouts:
        return cutouts[0].get('wcs', scene_wcs), tuple(scene_shape)
    return scene_wcs, tuple(scene_shape)


def _to_stack_grid(arr, arr_wcs, stack_wcs, stack_shape):
    """Reproject image to common stack grid (used only for stacked diagnostics)."""
    if stack_wcs is None or stack_shape is None:
        return np.asarray(arr, dtype=float)
    try:
        out, _ = reproject_interp((np.asarray(arr, dtype=float), arr_wcs), stack_wcs, shape_out=stack_shape)
        return np.nan_to_num(out, nan=np.nan)
    except Exception:
        return np.asarray(arr, dtype=float)


def _template_intersection_mask(cutouts, tpl_indices, stretch_mask, scene_shape):
    vm = np.ones(scene_shape, dtype=bool)
    if stretch_mask is not None and stretch_mask.shape == scene_shape:
        vm &= stretch_mask
    for i in tpl_indices:
        vm &= _bcd_valid_mask(cutouts[i], stretch_mask)
    return vm


def plot_template_component_stacks(cutouts, results, stars, star_fluxes, stretch_mask=None):
    """
    Template BCDs only: median data, submodel predictions, and median(data − submodel).
    """
    tpl = _template_cutout_indices(cutouts)
    if not tpl:
        print("   [Diagnostics] No template cutouts; skip DIAGNOSTIC_TEMPLATE_COMPONENTS.pdf")
        return
    scene_shape = results['scene_shape']
    vm = _template_intersection_mask(cutouts, tpl, stretch_mask, scene_shape)
    stack_wcs, stack_shape = _stack_grid(results, cutouts)

    def _med_stack(arrays):
        return np.nanmedian(np.stack(arrays, axis=0), axis=0)

    data_med = _med_stack([
        _to_stack_grid(np.asarray(cutouts[i]['data'], dtype=float), cutouts[i]['wcs'], stack_wcs, stack_shape)
        for i in tpl
    ])

    def _pred_med(**kw):
        return _med_stack([
            _to_stack_grid(
                solver.predict_cutout_model(
                    results, cutouts, stars, star_fluxes, i, **kw,
                ),
                cutouts[i]['wcs'],
                stack_wcs,
                stack_shape,
            )
            for i in tpl
        ])

    def _resid_med(**kw):
        return _med_stack([
            _to_stack_grid(
                np.asarray(cutouts[i]['data'], dtype=float)
                - solver.predict_cutout_model(results, cutouts, stars, star_fluxes, i, **kw),
                cutouts[i]['wcs'],
                stack_wcs,
                stack_shape,
            )
            for i in tpl
        ])

    panels = [
        ('median data', data_med, 'gray', 'flux'),
        (
            'median full pred',
            _pred_med(
                include_transient=True, include_stars=True,
                include_gp=True, include_host=True,
            ),
            'gray',
            'flux',
        ),
        (
            'median data − full',
            _resid_med(
                include_transient=True, include_stars=True,
                include_gp=True, include_host=True,
            ),
            'RdBu_r',
            'resid',
        ),
        (
            'median GP+BG',
            _pred_med(
                include_transient=False, include_stars=False,
                include_gp=True, include_host=False,
            ),
            'gray',
            'flux',
        ),
        (
            'median data − GP+BG',
            _resid_med(
                include_transient=False, include_stars=False,
                include_gp=True, include_host=False,
            ),
            'RdBu_r',
            'resid',
        ),
        (
            'median stars+BG',
            _pred_med(
                include_transient=False, include_stars=True,
                include_gp=False, include_host=False,
            ),
            'gray',
            'flux',
        ),
        (
            'median data − (stars+BG)',
            _resid_med(
                include_transient=False, include_stars=True,
                include_gp=False, include_host=False,
            ),
            'RdBu_r',
            'resid',
        ),
    ]
    if getattr(config, 'USE_HOST_GAUSSIAN_CORE', False):
        panels.extend([
            (
                'median host+BG',
                _pred_med(
                    include_transient=False, include_stars=False,
                    include_gp=False, include_host=True,
                ),
                'gray',
                'flux',
            ),
            (
                'median data − (host+BG)',
                _resid_med(
                    include_transient=False, include_stars=False,
                    include_gp=False, include_host=True,
                ),
                'RdBu_r',
                'resid',
            ),
        ])
    panels.extend([
        (
            'median GP+host+BG',
            _pred_med(
                include_transient=False, include_stars=False,
                include_gp=True, include_host=True,
            ),
            'gray',
            'flux',
        ),
        (
            'median data − (GP+host+BG)',
            _resid_med(
                include_transient=False, include_stars=False,
                include_gp=True, include_host=True,
            ),
            'RdBu_r',
            'resid',
        ),
    ])

    print("   [Diagnostics] Generating Template component stacks PDF...")
    pdf_path = os.path.join(config.DIAGNOSTIC_DIR, 'DIAGNOSTIC_TEMPLATE_COMPONENTS.pdf')
    with PdfPages(pdf_path) as pdf:
        for start in range(0, len(panels), 6):
            chunk = panels[start:start + 6]
            fig, axes = plt.subplots(2, 3, figsize=(14, 9))
            for ax, item in zip(axes.ravel(), chunk):
                title, arr, cmap, kind = item
                if kind == 'resid':
                    vmin, vmax = _residual_symmetric_limits(arr, mask=vm)
                    ax.imshow(arr, origin='lower', vmin=vmin, vmax=vmax, cmap=cmap)
                else:
                    _imshow_flux_asinh(
                        ax, arr, mask=vm, cmap=cmap,
                        pmin=1, pmax=95, width_pmin=1, width_pmax=95,
                    )
                ax.set_title(title, fontsize=9)
                ax.axis('off')
            for ax in axes.ravel()[len(chunk):]:
                ax.axis('off')
            plt.suptitle(f'Template stacks (n={len(tpl)} BCDs); p_lo–p_hi display', fontsize=11)
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close()
    print(f"   [Diagnostics] Wrote {pdf_path}")


def plot_template_prf_vs_residual_orientation(
    cutouts, results, stars, star_fluxes, stretch_mask=None,
):
    """
    Median normalized PRF on the analysis grid vs median (data − full pred), template BCDs only.
    """
    tpl = _template_cutout_indices(cutouts)
    if not tpl:
        print("   [Diagnostics] No template cutouts; skip PRF vs residual orientation figure")
        return
    scene_wcs = results['scene_wcs']
    scene_shape = results['scene_shape']
    stack_wcs, stack_shape = _stack_grid(results, cutouts)
    vm = _template_intersection_mask(cutouts, tpl, stretch_mask, scene_shape)

    ra_h = getattr(config, 'HOST_CORE_RA', None)
    dec_h = getattr(config, 'HOST_CORE_DEC', None)
    if ra_h is not None and dec_h is not None:
        ra, dec = float(ra_h), float(dec_h)
    else:
        ra = float(config.TRANSIENT_RA) + float(results.get('transient_dra_deg', 0.0))
        dec = float(config.TRANSIENT_DEC) + float(results.get('transient_ddec_deg', 0.0))

    prf_cube = []
    resid_cube = []
    for i in tpl:
        c = cutouts[i]
        chan = 'ch2' if 'ch2' in str(c.get('filename', '')).lower() else 'ch1'
        w_native = c['raw_wcs']
        tx, ty = w_native.world_to_pixel_values(ra, dec)
        prf_m = solver.load_prf(chan, tx, ty)
        col = solver.generate_prf_fast(
            scene_wcs, w_native, prf_m, ra, dec, scene_shape,
            channel=chan, is_full_array=c.get('is_full_array', False),
        )
        prf_cube.append(_to_stack_grid(col.reshape(scene_shape), scene_wcs, stack_wcs, stack_shape))
        pred = solver.predict_cutout_model(
            results, cutouts, stars, star_fluxes, i,
            include_transient=True, include_stars=True,
        )
        resid_cube.append(
            _to_stack_grid(np.asarray(c['data'], dtype=float) - pred, c['wcs'], stack_wcs, stack_shape)
        )

    med_prf = np.nanmedian(np.stack(prf_cube, axis=0), axis=0)
    med_res = np.nanmedian(np.stack(resid_cube, axis=0), axis=0)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8))
    _imshow_flux_asinh(
        axes[0], med_prf, mask=vm, cmap='gray',
        pmin=1, pmax=95, width_pmin=1, width_pmax=95,
    )
    axes[0].set_title('median PRF (norm.) template epochs')
    rlo, rhi = _residual_symmetric_limits(med_res, mask=vm)
    axes[1].imshow(med_res, origin='lower', vmin=rlo, vmax=rhi, cmap='RdBu_r')
    axes[1].set_title('median resid (data−full)')
    for ax in axes:
        ax.axis('off')
    plt.suptitle(
        'Template epochs: PRF (gray) and residual (RdBu); PRF scaled p_lo–p_hi on combined sample',
        fontsize=10,
    )
    plt.tight_layout()
    out_path = os.path.join(config.DIAGNOSTIC_DIR, 'DIAGNOSTIC_TEMPLATE_PRF_VS_RESID.png')
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"   [Diagnostics] Wrote {out_path}")


def run_template_dipole_chi2_refinement(cutouts, results, stars, star_fluxes, stretch_mask=None):
    """
    Template BCDs: dipole χ² scan (see src/dipole_chi2_scan.py).
    """
    from . import dipole_chi2_scan

    return dipole_chi2_scan.compute_dipole_chi2_scan(
        cutouts, results, stars, star_fluxes, stretch_mask,
    )


def plot_gp_vs_stars(cutouts, results, stars, star_fluxes, stretch_mask=None):
    """
    Compare GP+BG vs star PRFs+BG on one science frame (matched asinh stretch).
    """
    sci_idx = [i for i, c in enumerate(cutouts) if not c.get('is_template')]
    if not sci_idx:
        return
    i = sci_idx[0]
    gp_bg = solver.predict_cutout_model(
        results, cutouts, stars, star_fluxes, i,
        include_gp=True, include_transient=False, include_stars=False, include_host=True,
    )
    stars_bg = solver.predict_cutout_model(
        results, cutouts, stars, star_fluxes, i,
        include_gp=False, include_transient=False, include_stars=True, include_host=True,
    )
    vm = _bcd_valid_mask(cutouts[i], stretch_mask)
    lw = max(_asinh_linear_width(gp_bg, mask=vm), _asinh_linear_width(stars_bg, mask=vm))
    lo = min(
        get_robust_limits(gp_bg, 1, 99, mask=vm)[0],
        get_robust_limits(stars_bg, 1, 99, mask=vm)[0],
    )
    hi = max(
        get_robust_limits(gp_bg, 1, 99, mask=vm)[1],
        get_robust_limits(stars_bg, 1, 99, mask=vm)[1],
    )
    norm = AsinhNorm(linear_width=lw, vmin=lo, vmax=hi)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    axes[0].imshow(gp_bg, origin='lower', cmap='gray', norm=norm, interpolation='nearest')
    axes[0].set_title('GP scene + BG (asinh)')
    axes[1].imshow(stars_bg, origin='lower', cmap='gray', norm=norm, interpolation='nearest')
    axes[1].set_title('Star PRFs + BG (asinh)')
    for ax in axes:
        ax.axis('off')
    plt.suptitle(
        f'GP vs stars — matched asinh (frame {i}, {cutouts[i].get("filename", "")[-28:]})',
        fontsize=11,
    )
    plt.tight_layout()
    plt.savefig(os.path.join(config.DIAGNOSTIC_DIR, 'DIAGNOSTIC_GP_VS_STARS.png'), dpi=150)
    plt.close()


def plot_lightcurve(table, chan_str):
    plt.figure(figsize=(12, 6))
    mask_tpl = (table['Is_Template'] == 1)
    if np.sum(mask_tpl) > 0:
        ph = []
        for j in np.where(mask_tpl)[0]:
            if _lightcurve_row_is_placeholder(
                table['Flux_Jy'][j], table['Flux_Err_Jy'][j], True
            ):
                continue
            ph.append(j)
        if ph:
            ph = np.array(ph, dtype=int)
            plt.errorbar(
                table['MJD'][ph], table['Flux_Jy'][ph], yerr=table['Flux_Err_Jy'][ph],
                fmt='o', color='gray', label='Template', alpha=0.5,
            )
    mask_sci = (table['Is_Template'] == 0)
    if np.sum(mask_sci) > 0:
        plt.errorbar(table['MJD'][mask_sci], table['Flux_Jy'][mask_sci], yerr=table['Flux_Err_Jy'][mask_sci], fmt='o', color='blue', label='Science')
        
    epoch_ids = np.unique(table['Epoch_ID'])
    bin_mjd, bin_flux, bin_err = [], [], []
    for ep in epoch_ids:
        mask = (table['Epoch_ID'] == ep) & (table['Is_Template'] == 0)
        if np.sum(mask) > 0:
            fluxes = table['Flux_Jy'][mask]
            errs = np.asarray(table['Flux_Err_Jy'][mask], dtype=float)
            w = 1.0 / (np.maximum(errs, 1e-30) ** 2)
            avg = np.sum(fluxes * w) / np.sum(w)
            bin_mjd.append(np.median(table['MJD'][mask]))
            bin_flux.append(avg)
            bin_err.append(np.sqrt(1.0/np.sum(w)))
            
    if bin_mjd:
        plt.errorbar(
            bin_mjd, bin_flux, yerr=bin_err, fmt='s', color='red', markersize=8,
            markeredgecolor='k', label='Epoch mean',
        )

    plt.xlabel("MJD")
    plt.ylabel("Flux (Jy)")
    plt.title(f"Light curve ({chan_str}) — per-BCD (epoch flux repeated)")
    plt.legend(loc='best', fontsize=9)
    plt.grid(True, alpha=0.35, which='both', linestyle='-', linewidth=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(config.OUTPUT_DIR, f'lightcurve_{chan_str}.png'), dpi=150)
    plt.close()


def _json_sanitize(obj):
    """Replace non-finite floats so fit_quality JSON is strict-compliant."""
    if isinstance(obj, dict):
        return {k: _json_sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_sanitize(v) for v in obj]
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, np.floating):
        x = float(obj)
        return None if math.isnan(x) or math.isinf(x) else x
    if isinstance(obj, np.integer):
        return int(obj)
    return obj


def _template_submodel_rmse_medians(cutouts, results, stars, star_fluxes, stretch_mask, annulus_mask=None):
    """Median per-BCD RMSE (Jy) over template cutouts for each prediction mode."""
    tpl_ix = _template_cutout_indices(cutouts)
    if not tpl_ix:
        return {}
    specs = [
        ('full', dict(
            include_transient=True, include_stars=True, include_gp=True, include_host=True,
        )),
        ('gp_bg', dict(
            include_transient=False, include_stars=False, include_gp=True, include_host=False,
        )),
        ('stars_bg', dict(
            include_transient=False, include_stars=True, include_gp=False, include_host=False,
        )),
        ('host_bg', dict(
            include_transient=False, include_stars=False, include_gp=False, include_host=True,
        )),
        ('gp_host_bg', dict(
            include_transient=False, include_stars=False, include_gp=True, include_host=True,
        )),
    ]
    out = {}
    for name, kw in specs:
        vals = []
        for i in tpl_ix:
            c = cutouts[i]
            d = np.asarray(c['data'], dtype=float)
            pred = solver.predict_cutout_model(
                results, cutouts, stars, star_fluxes, i, **kw,
            )
            r = d - pred
            m = _bcd_valid_mask(c, stretch_mask)
            if annulus_mask is not None and annulus_mask.shape == m.shape:
                m &= annulus_mask
            if np.sum(m) > 4:
                vals.append(float(np.sqrt(np.mean(r[m] ** 2))))
        if vals:
            out[name] = float(np.median(vals))
    return out


def write_fit_quality_report(cutouts, results, stars, star_fluxes, median_stamp, chan_str):
    """
    Write JSON + short text summary: residual RMS per frame, template transient constraint,
    and uncertainty summaries for QA.
    """
    os.makedirs(config.DIAGNOSTIC_DIR, exist_ok=True)
    stretch_mask = diagnostic_stretch_mask(cutouts, median_stamp.shape)
    model = np.asarray(results['model_scene'])
    gp_scene = np.asarray(results.get('gp_scene', model))
    bgs = results['epoch_backgrounds']
    tflux = np.asarray(results['transient_fluxes'])
    terr = np.asarray(results.get('transient_errs', np.zeros(len(cutouts))))
    serr = np.asarray(results.get('star_errs', np.zeros(len(stars))))

    inner_r = float(getattr(config, 'SUPERRES_QA_INNER_MASK_PX', 3.0))
    outer_r = float(getattr(config, 'SUPERRES_QA_OUTER_RADIUS_PX', 12.0))
    frac_max = float(getattr(config, 'SUPERRES_QA_POISSON_FRACTION_MAX', 0.10))

    scene_wcs = results.get('scene_wcs')
    scene_shape = results.get('scene_shape')
    prf_acf_e90 = None
    if scene_wcs is not None and scene_shape is not None:
        for c in cutouts:
            if c.get('is_template'):
                continue
            ch = 'ch2' if 'ch2' in str(c.get('filename', '')).lower() else 'ch1'
            try:
                prf_acf_e90 = residual_metrics.prf_autocorr_scale_on_grid(
                    scene_wcs,
                    c['raw_wcs'],
                    ch,
                    float(config.TRANSIENT_RA),
                    float(config.TRANSIENT_DEC),
                    scene_shape,
                    bool(c.get('is_full_array', False)),
                )
            except Exception:
                prf_acf_e90 = None
            break

    per_frame = []
    for i, c in enumerate(cutouts):
        d = np.asarray(c['data'])
        sig = np.asarray(c['sigma'])
        pred = solver.predict_cutout_model(
            results, cutouts, stars, star_fluxes, i,
            include_transient=True, include_stars=True,
        )
        mask = (d != 0) & np.isfinite(sig) & (sig < 1e20)
        if np.sum(mask) == 0:
            continue
        resid = d - pred
        rmse = float(np.sqrt(np.nanmean(resid[mask] ** 2)))
        mad = float(np.nanmedian(np.abs(resid[mask])))
        resid_struct = _json_sanitize(
            residual_metrics.summarize_frame_residual(resid, sig, mask)
        )
        h, w = d.shape
        cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
        yy, xx = np.mgrid[0:h, 0:w]
        rpix = np.hypot(xx - cx, yy - cy)
        annulus = mask & (rpix >= inner_r) & (rpix <= outer_r)
        ann_rmse = float(np.sqrt(np.nanmean(resid[annulus] ** 2))) if np.any(annulus) else None
        med_sig_ann = float(np.nanmedian(sig[annulus])) if np.any(annulus) else None
        pois_ann = float(np.sqrt(max(np.nanmedian(np.abs(d[annulus])), 0.0))) if np.any(annulus) else None
        ref_noise = None
        if med_sig_ann is not None and np.isfinite(med_sig_ann) and med_sig_ann > 0:
            ref_noise = med_sig_ann
        elif pois_ann is not None and np.isfinite(pois_ann) and pois_ann > 0:
            ref_noise = pois_ann
        passes_ann = bool(
            ann_rmse is not None and ref_noise is not None and ref_noise > 0
            and ann_rmse < frac_max * ref_noise
        )
        row = {
            'index': i,
            'filename': c.get('filename', ''),
            'is_template': bool(c.get('is_template', False)),
            'rmse_jy_fullpred': rmse,
            'mad_jy_fullpred': mad,
            'annulus_rmse_jy': ann_rmse,
            'annulus_median_sigma_jy': med_sig_ann,
            'annulus_poisson_proxy_jy': pois_ann,
            'superres_annulus_passes_fraction': passes_ann,
            'transient_flux': float(tflux[i]),
            'transient_err': float(terr[i]) if i < len(terr) else 0.0,
        }
        if resid_struct:
            row['residual_structure'] = resid_struct
        per_frame.append(row)

    tpl_mask = np.array([bool(c.get('is_template', False)) for c in cutouts])
    sci_mask = ~tpl_mask
    tpl_trans_rms = float(np.sqrt(np.mean(tflux[tpl_mask] ** 2))) if np.any(tpl_mask) else 0.0
    tpl_trans_max = float(np.max(np.abs(tflux[tpl_mask]))) if np.any(tpl_mask) else 0.0

    dra = float(results.get('transient_dra_deg', 0.0))
    dde = float(results.get('transient_ddec_deg', 0.0))
    cosdec = np.cos(np.deg2rad(float(config.TRANSIENT_DEC)))
    offset_as = float(3600.0 * np.hypot(dra * cosdec, dde))

    scene_wcs_r = results.get('scene_wcs')
    scene_shape_r = results.get('scene_shape')
    g_ann = None
    if scene_wcs_r is not None and scene_shape_r is not None:
        g_ann = galaxy_disk_annulus_mask(scene_wcs_r, scene_shape_r)
    template_rmse = _template_submodel_rmse_medians(
        cutouts, results, stars, star_fluxes, stretch_mask, annulus_mask=None,
    )
    template_rmse_disk = _template_submodel_rmse_medians(
        cutouts, results, stars, star_fluxes, stretch_mask, annulus_mask=g_ann,
    )
    dipole_pack = run_template_dipole_chi2_refinement(
        cutouts, results, stars, star_fluxes, stretch_mask,
    )

    sci_eids = results.get('science_epoch_ids')
    if sci_eids is not None:
        sci_eids = np.asarray(sci_eids).tolist()
    te_flux = results.get('transient_epoch_fluxes')
    if te_flux is not None:
        te_flux = np.asarray(te_flux, dtype=float).tolist()
    te_err = results.get('transient_epoch_errs')
    if te_err is not None:
        te_err = np.asarray(te_err, dtype=float).tolist()

    report = {
        'channel': chan_str,
        'n_cutouts': len(cutouts),
        'n_stars': len(stars),
        'host_core_flux': float(results.get('host_core_flux', 0.0)),
        'host_core_err': float(results.get('host_core_err', 0.0)),
        'prf_acf_e90_scale_pix': prf_acf_e90,
        'science_epoch_ids': sci_eids,
        'transient_epoch_fluxes': te_flux,
        'transient_epoch_errs': te_err,
        'transient_bg_cov_by_epoch_id': results.get('transient_bg_cov_by_epoch_id', {}),
        'transient_dra_deg': dra,
        'transient_ddec_deg': dde,
        'transient_dra_err_deg': float(results.get('transient_dra_err_deg', 0.0)),
        'transient_ddec_err_deg': float(results.get('transient_ddec_err_deg', 0.0)),
        'transient_offset_great_circle_arcsec': offset_as,
        'template_transient_rms': tpl_trans_rms,
        'template_transient_max_abs': tpl_trans_max,
        'science_transient_err_median': float(np.median(terr[sci_mask])) if np.any(sci_mask) else 0.0,
        'star_flux_median': float(np.median(star_fluxes)) if len(star_fluxes) else 0.0,
        'star_err_median': float(np.median(serr)) if len(serr) else 0.0,
        'per_frame': per_frame,
        'median_stamp_center_vs_recon_model': float(
            np.nanmedian(median_stamp - model) if median_stamp.shape == model.shape else np.nan
        ),
        'median_stamp_center_vs_gp_scene': float(
            np.nanmedian(median_stamp - gp_scene) if median_stamp.shape == gp_scene.shape else np.nan
        ),
        'superres_qa_inner_mask_px': inner_r,
        'superres_qa_outer_radius_px': outer_r,
        'superres_qa_fraction_max': frac_max,
        'galaxy_extended_center_ra': getattr(config, 'GALAXY_EXTENDED_CENTER_RA', None),
        'galaxy_extended_center_dec': getattr(config, 'GALAXY_EXTENDED_CENTER_DEC', None),
        'template_median_rmse_jy_by_submodel': _json_sanitize(template_rmse),
        'template_median_rmse_jy_disk_annulus': _json_sanitize(template_rmse_disk),
        'template_dipole_chi2_refinement': _json_sanitize(dipole_pack.get('dipole_chi2_refinement', {})),
        'diagnostic_display': {
            'flux_panels': 'AsinhNorm',
            'bcd_flux_percentiles': list(_bcd_flux_percentiles()),
            'imshow_linear_percentiles_lo_hi': [
                float(getattr(config, 'DIAGNOSTIC_IMSHOW_PERCENTILES_LO', 1.0)),
                float(getattr(config, 'DIAGNOSTIC_IMSHOW_PERCENTILES_HI', 95.0)),
            ],
            'bcd_residuals_row2': 'per-pixel residual/sigma',
            'epoch_stacks': 'median Jy + median sigma pages per epoch',
            'fit_template': 'two-panel: template + transient-only (no alpha blend)',
            'asinh_width_frac': float(getattr(config, 'DIAGNOSTIC_ASINH_WIDTH_FRAC', 0.12)),
            'resid_sigma_cap': float(getattr(config, 'DIAGNOSTIC_RESID_SIGMA_DISPLAY_CAP', 6.0)),
        },
    }

    json_path = os.path.join(config.DIAGNOSTIC_DIR, f'fit_quality_{chan_str}.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)

    txt_path = os.path.join(config.DIAGNOSTIC_DIR, f'fit_quality_{chan_str}.txt')
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("Fit quality summary\n")
        f.write(f"  template transient RMS (should be ~0): {tpl_trans_rms:.3e}\n")
        f.write(f"  template max |transient|: {tpl_trans_max:.3e}\n")
        f.write(f"  science median flux uncertainty: {report['science_transient_err_median']:.3e}\n")
        if te_flux is not None and sci_eids is not None and len(te_flux) == len(sci_eids):
            f.write("  transient flux by science epoch_id (Jy):\n")
            terr_ep = (
                te_err if te_err is not None and len(te_err) == len(te_flux)
                else [0.0] * len(te_flux)
            )
            for eid, fval, ferr in zip(sci_eids, te_flux, terr_ep):
                f.write(f"    epoch {eid}: {fval:.3e} ± {float(ferr):.3e}\n")
        f.write(
            f"  fitted transient offset: dRA={dra:.3e} deg, dDec={dde:.3e} deg "
            f"(|Δ|≈{offset_as:.4f} arcsec)\n"
        )
        f.write(f"  median |stamp - recon model|: {report['median_stamp_center_vs_recon_model']:.3e}\n")
        f.write(f"  median |stamp - gp_scene|: {report['median_stamp_center_vs_gp_scene']:.3e}\n")
        n_pass = sum(1 for p in per_frame if p.get('superres_annulus_passes_fraction'))
        f.write(f"  annulus super-res checks passed (frames): {n_pass}/{len(per_frame)}\n")
    print(f"   [Diagnostics] Wrote {json_path}")
