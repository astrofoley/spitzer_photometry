import os

import matplotlib.pyplot as plt
import numpy as np
import pytest
from astropy.wcs import WCS
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import AsinhNorm

from src import config, solver


def _rot_wcs(n_pix: int, ra: float, dec: float, theta_deg: float, pixel_scale_arcsec: float) -> WCS:
    w = WCS(naxis=2)
    w.wcs.crpix = [n_pix / 2, n_pix / 2]
    w.wcs.crval = [ra, dec]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    scale = float(pixel_scale_arcsec) / 3600.0
    w.wcs.cdelt = [-scale, scale]
    th = np.deg2rad(theta_deg)
    c, s = np.cos(th), np.sin(th)
    w.wcs.pc = np.array([[c, -s], [s, c]])
    return w


def _scene_wcs_from_bcd_footprints(
    bcd_wcs_list,
    bcd_shape,
    scene_pixel_scale_arcsec: float,
    sky_points_deg,
    pad_scene_px: int = 6,
):
    """
    Build a North-up scene WCS/shape that covers all BCD corners and required sky points.
    """
    ra_all = []
    dec_all = []
    h, w = int(bcd_shape[0]), int(bcd_shape[1])
    corners = np.array(
        [[0.0, 0.0], [w - 1.0, 0.0], [0.0, h - 1.0], [w - 1.0, h - 1.0]],
        dtype=float,
    )
    for wb in bcd_wcs_list:
        ra_c, dec_c = wb.pixel_to_world_values(corners[:, 0], corners[:, 1])
        ra_all.extend(np.asarray(ra_c, dtype=float).tolist())
        dec_all.extend(np.asarray(dec_c, dtype=float).tolist())
    for ra_p, dec_p in sky_points_deg:
        ra_all.append(float(ra_p))
        dec_all.append(float(dec_p))

    ra_ref = float(np.mean(ra_all))
    dec_ref = float(np.mean(dec_all))
    cosd = max(np.cos(np.deg2rad(dec_ref)), 1e-6)
    x_arcsec = (np.asarray(ra_all) - ra_ref) * cosd * 3600.0
    y_arcsec = (np.asarray(dec_all) - dec_ref) * 3600.0
    pix_scale = float(scene_pixel_scale_arcsec)
    x_pix = x_arcsec / pix_scale
    y_pix = y_arcsec / pix_scale
    x_min = float(np.min(x_pix)) - float(pad_scene_px)
    x_max = float(np.max(x_pix)) + float(pad_scene_px)
    y_min = float(np.min(y_pix)) - float(pad_scene_px)
    y_max = float(np.max(y_pix)) + float(pad_scene_px)
    w_scene = int(np.ceil(x_max - x_min + 1.0))
    h_scene = int(np.ceil(y_max - y_min + 1.0))
    crpix1 = 1.0 - x_min
    crpix2 = 1.0 - y_min

    w_scene_wcs = WCS(naxis=2)
    w_scene_wcs.wcs.crpix = [crpix1, crpix2]
    w_scene_wcs.wcs.crval = [ra_ref, dec_ref]
    w_scene_wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w_scene_wcs.wcs.cdelt = [-pix_scale / 3600.0, pix_scale / 3600.0]
    w_scene_wcs.wcs.pc = np.eye(2)
    return w_scene_wcs, (h_scene, w_scene)


def _scene_wcs_budgeted(
    bcd_wcs_list,
    bcd_shape,
    min_scene_pixel_scale_arcsec: float,
    sky_points_deg,
    max_scene_pixels: int,
    pad_scene_px: int = 6,
):
    """
    Choose finest scene pixel scale that still keeps total scene pixels <= max_scene_pixels.
    """
    scale = float(min_scene_pixel_scale_arcsec)
    for _ in range(20):
        w_scene, shp = _scene_wcs_from_bcd_footprints(
            bcd_wcs_list,
            bcd_shape,
            scene_pixel_scale_arcsec=scale,
            sky_points_deg=sky_points_deg,
            pad_scene_px=pad_scene_px,
        )
        if int(shp[0] * shp[1]) <= int(max_scene_pixels):
            return w_scene, shp, scale
        scale *= 1.12
    return w_scene, shp, scale


def _percentile_limits(arr: np.ndarray, lo: float = 1.0, hi: float = 95.0):
    v = np.asarray(arr, dtype=float)
    vv = v[np.isfinite(v)]
    if vv.size < 4:
        return 0.0, 1.0
    a, b = np.percentile(vv, [lo, hi])
    if b <= a:
        b = a + 1e-12
    return float(a), float(b)


def _bcd_asinh_norm(data: np.ndarray):
    """Asinh stretch for BCD-like panels: vmin/vmax from 1-99.99% of data."""
    lo, hi = _percentile_limits(data, lo=1.0, hi=99.99)
    pw_lo, pw_hi = _percentile_limits(data, lo=0.5, hi=99.99)
    span = max(pw_hi - pw_lo, 1e-30)
    lw = max(float(getattr(config, "DIAGNOSTIC_ASINH_WIDTH_FRAC", 0.12)) * span, 1e-20)
    return AsinhNorm(linear_width=lw, vmin=lo, vmax=hi)


def _superres(arr: np.ndarray, upsample: int = 10) -> np.ndarray:
    up = max(2, int(upsample))
    a = np.asarray(arr, dtype=float)
    return np.repeat(np.repeat(a, up, axis=0), up, axis=1)


def _to_superres_pixel(x: float, y: float, upsample: int):
    up = float(max(2, int(upsample)))
    return (float(x) + 0.5) * up - 0.5, (float(y) + 0.5) * up - 0.5


def _compass_vectors(w: WCS, ra_deg: float, dec_deg: float):
    eps_dec = 1.0 / 3600.0
    cosd = max(np.cos(np.deg2rad(dec_deg)), 1e-6)
    eps_ra = eps_dec / cosd
    x0, y0 = w.world_to_pixel_values(ra_deg, dec_deg)
    x_n, y_n = w.world_to_pixel_values(ra_deg, dec_deg + eps_dec)
    x_e, y_e = w.world_to_pixel_values(ra_deg + eps_ra, dec_deg)
    return (float(x_n - x0), float(y_n - y0)), (float(x_e - x0), float(y_e - y0))


def _draw_marker_and_compass(ax, x: float, y: float, v_n, v_e, shape):
    ax.plot([x], [y], marker="x", ms=7, mew=1.6, color="lime")
    h, w = shape
    base_x = 0.84 * (w - 1)
    base_y = 0.11 * (h - 1)
    L = 0.09 * min(h, w)

    def _unit(vx, vy):
        nrm = np.hypot(vx, vy)
        if nrm <= 1e-12:
            return 0.0, 1.0
        return vx / nrm, vy / nrm

    nux, nuy = _unit(*v_n)
    eux, euy = _unit(*v_e)
    ax.annotate("", xy=(base_x + L * nux, base_y + L * nuy), xytext=(base_x, base_y), arrowprops=dict(color="yellow", lw=1.6))
    ax.annotate("", xy=(base_x + L * eux, base_y + L * euy), xytext=(base_x, base_y), arrowprops=dict(color="cyan", lw=1.6))
    ax.text(base_x + 1.08 * L * nux, base_y + 1.08 * L * nuy, "N", color="yellow", fontsize=8, ha="center", va="center")
    ax.text(base_x + 1.08 * L * eux, base_y + 1.08 * L * euy, "E", color="cyan", fontsize=8, ha="center", va="center")


def _resid_sym_limits(arr: np.ndarray):
    plo = float(getattr(config, "DIAGNOSTIC_RESIDUAL_PERCENTILES_LO", 1.0))
    phi = float(getattr(config, "DIAGNOSTIC_RESIDUAL_PERCENTILES_HI", 99.0))
    v = np.asarray(arr, dtype=float)
    vv = v[np.isfinite(v)]
    if vv.size < 4:
        return -1.0, 1.0
    lo, hi = np.percentile(vv, [plo, phi])
    lim = max(abs(float(lo)), abs(float(hi)), 1e-12)
    return -lim, lim


def _intrinsic_components(results):
    """Intrinsic components on scene grid (pre-PRF, no BG, no stars/transient)."""
    scene_wcs = results["scene_wcs"]
    scene_shape = results["scene_shape"]
    gp = np.asarray(results.get("gp_scene", results["model_scene"]), dtype=float)
    host = np.zeros(scene_shape, dtype=float)
    nps = np.zeros(scene_shape, dtype=float)

    if getattr(config, "USE_HOST_GAUSSIAN_CORE", False):
        ra_h = getattr(config, "HOST_CORE_RA", None)
        dec_h = getattr(config, "HOST_CORE_DEC", None)
        if ra_h is not None and dec_h is not None:
            col_h = solver.host_core_gaussian_column(
                scene_wcs,
                float(ra_h),
                float(dec_h),
                float(getattr(config, "HOST_CORE_SIGMA_PX", 1.5)),
                scene_shape,
            )
            host += float(results.get("host_core_flux", 0.0)) * col_h.reshape(scene_shape)

    if getattr(config, "USE_NUCLEAR_POINT_SOURCE", False):
        ra_np = getattr(config, "NUCLEAR_POINT_RA", None)
        dec_np = getattr(config, "NUCLEAR_POINT_DEC", None)
        if ra_np is None or dec_np is None:
            ra_np = getattr(config, "HOST_CORE_RA", None)
            dec_np = getattr(config, "HOST_CORE_DEC", None)
        if ra_np is not None and dec_np is not None:
            fnp = float(results.get("nuclear_point_flux", 0.0))
            nx, ny = scene_wcs.world_to_pixel_values(float(ra_np), float(dec_np))
            solver._add_delta_to_image(nps, float(nx), float(ny), fnp)
    return gp, host, nps


def _synthetic_galaxy_scene(shape, cx: float, cy: float):
    h, w = int(shape[0]), int(shape[1])
    yy, xx = np.mgrid[0:h, 0:w].astype(float)
    r2 = (xx - cx) ** 2 + (yy - cy) ** 2
    core = 2.4e-5 * np.exp(-0.5 * r2 / (2.6 ** 2))
    bulge = 1.8e-5 * np.exp(-0.5 * r2 / (6.0 ** 2))
    return core + bulge


def _write_native_fit_pdf(n_bcd: int, cutouts, results):
    os.makedirs(config.DIAGNOSTIC_DIR, exist_ok=True)
    out_pdf = os.path.join(config.DIAGNOSTIC_DIR, f"NATIVE_FIT_DIAGNOSTIC_N{n_bcd}.pdf")
    up = 40
    gp_i, host_i, nps_i = _intrinsic_components(results)
    with PdfPages(out_pdf) as pdf:
        for i, c in enumerate(cutouts):
            data = np.asarray(c["data"], dtype=float)
            bg_i = float(np.asarray(results.get("bcd_backgrounds", np.zeros(len(cutouts))))[i])

            pred = solver.predict_cutout_model(
                results,
                cutouts,
                [],
                [],
                i,
                include_gp=True,
                include_transient=True,
                include_stars=False,
                include_host=True,
                include_nuclear_point=True,
            )
            resid = data - pred

            flux_norm_bcd = _bcd_asinh_norm(data)

            gp_hi = _superres(gp_i, upsample=up) + bg_i
            host_hi = _superres(host_i, upsample=up) + bg_i
            scene_wcs = results["scene_wcs"]
            ra_n = float(getattr(config, "NUCLEAR_POINT_RA", config.TRANSIENT_RA))
            dec_n = float(getattr(config, "NUCLEAR_POINT_DEC", config.TRANSIENT_DEC))
            npx, npy = scene_wcs.world_to_pixel_values(ra_n, dec_n)
            nps_single = np.zeros_like(nps_i)
            ix = int(np.clip(np.round(npx), 0, nps_single.shape[1] - 1))
            iy = int(np.clip(np.round(npy), 0, nps_single.shape[0] - 1))
            nps_single[iy, ix] = float(results.get("nuclear_point_flux", 0.0))
            nps_hi = _superres(nps_single, upsample=up) + bg_i
            full_hi = _superres(gp_i + host_i + nps_i, upsample=up) + bg_i
            flux_norm_hi = _bcd_asinh_norm(full_hi)

            fig, axes = plt.subplots(2, 4, figsize=(18, 9))
            ax = axes.ravel()

            im0 = ax[0].imshow(data, origin="lower", cmap="gray", norm=flux_norm_bcd, interpolation="nearest")
            ax[0].set_title("BCD (unaltered); Asinh 1-99.99%")
            plt.colorbar(im0, ax=ax[0], fraction=0.046, pad=0.04)

            im1 = ax[1].imshow(gp_hi, origin="lower", cmap="gray", norm=flux_norm_hi, interpolation="nearest")
            ax[1].set_title("N-up super-res: GP only (+BG); no PRF (panel-matched)")
            plt.colorbar(im1, ax=ax[1], fraction=0.046, pad=0.04)

            im2 = ax[2].imshow(host_hi, origin="lower", cmap="gray", norm=flux_norm_hi, interpolation="nearest")
            ax[2].set_title("N-up super-res: host Gaussian (+BG); no PRF (panel-matched)")
            plt.colorbar(im2, ax=ax[2], fraction=0.046, pad=0.04)

            im3 = ax[3].imshow(nps_hi, origin="lower", cmap="gray", norm=flux_norm_hi, interpolation="nearest")
            ax[3].set_title("N-up super-res: nuclear delta (+BG); no PRF (panel-matched)")
            plt.colorbar(im3, ax=ax[3], fraction=0.046, pad=0.04)

            im4 = ax[4].imshow(full_hi, origin="lower", cmap="gray", norm=flux_norm_hi, interpolation="nearest")
            ax[4].set_title("N-up super-res: GP+host+delta+BG; no PRF (panel-matched)")
            plt.colorbar(im4, ax=ax[4], fraction=0.046, pad=0.04)

            im5 = ax[5].imshow(pred, origin="lower", cmap="gray", norm=flux_norm_bcd, interpolation="nearest")
            ax[5].set_title("Model on BCD (SV-PRF + project); same stretch as BCD")
            plt.colorbar(im5, ax=ax[5], fraction=0.046, pad=0.04)

            vmin, vmax = _resid_sym_limits(resid)
            im6 = ax[6].imshow(resid, origin="lower", cmap="RdBu_r", vmin=vmin, vmax=vmax, interpolation="nearest")
            ax[6].set_title("Residual (data - model); linear symmetric")
            plt.colorbar(im6, ax=ax[6], fraction=0.046, pad=0.04)

            ra_t = float(config.TRANSIENT_RA)
            dec_t = float(config.TRANSIENT_DEC)
            tx_b, ty_b = c["wcs"].world_to_pixel_values(ra_t, dec_t)
            v_n_b, v_e_b = _compass_vectors(c["wcs"], ra_t, dec_t)
            _draw_marker_and_compass(ax[0], tx_b, ty_b, v_n_b, v_e_b, data.shape)
            _draw_marker_and_compass(ax[5], tx_b, ty_b, v_n_b, v_e_b, pred.shape)
            _draw_marker_and_compass(ax[6], tx_b, ty_b, v_n_b, v_e_b, resid.shape)

            tx_s, ty_s = results["scene_wcs"].world_to_pixel_values(ra_t, dec_t)
            tx_h, ty_h = _to_superres_pixel(tx_s, ty_s, up)
            v_n_s, v_e_s = _compass_vectors(results["scene_wcs"], ra_t, dec_t)
            v_n_h = (v_n_s[0] * up, v_n_s[1] * up)
            v_e_h = (v_e_s[0] * up, v_e_s[1] * up)
            _draw_marker_and_compass(ax[1], tx_h, ty_h, v_n_h, v_e_h, gp_hi.shape)
            _draw_marker_and_compass(ax[2], tx_h, ty_h, v_n_h, v_e_h, host_hi.shape)
            _draw_marker_and_compass(ax[3], tx_h, ty_h, v_n_h, v_e_h, nps_hi.shape)
            _draw_marker_and_compass(ax[4], tx_h, ty_h, v_n_h, v_e_h, full_hi.shape)

            ax[7].set_visible(False)

            for j in range(7):
                ax[j].axis("off")
            fig.suptitle(
                f"Native template-fit diagnostics: N={n_bcd}, frame={i}, template={bool(c.get('is_template'))}",
                fontsize=11,
            )
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
    return out_pdf


@pytest.mark.slow
@pytest.mark.parametrize("n_bcd", [1, 2, 10])
def test_native_fit_runs_for_n_bcds(n_bcd: int):
    n_pix = 48
    ra0 = 197.450286
    dec0 = -23.381497
    rng = np.random.default_rng(123 + n_bcd)
    nuc_ra = 197.448762
    nuc_dec = -23.383962

    prev_host = (
        config.USE_HOST_GAUSSIAN_CORE,
        config.HOST_CORE_RA,
        config.HOST_CORE_DEC,
        config.HOST_CORE_SIGMA_PX,
        config.HOST_CORE_NONNEGATIVE,
    )
    prev_nps = (
        config.USE_NUCLEAR_POINT_SOURCE,
        config.NUCLEAR_POINT_RA,
        config.NUCLEAR_POINT_DEC,
        config.NUCLEAR_POINT_NONNEGATIVE,
    )
    prev_max_scene = config.MAX_SCENE_PIXELS
    prev_transient = (config.TRANSIENT_RA, config.TRANSIENT_DEC)
    config.TRANSIENT_RA = float(ra0)
    config.TRANSIENT_DEC = float(dec0)
    config.USE_HOST_GAUSSIAN_CORE = False
    config.HOST_CORE_RA = float(nuc_ra)
    config.HOST_CORE_DEC = float(nuc_dec)
    config.HOST_CORE_SIGMA_PX = 1.7
    config.HOST_CORE_NONNEGATIVE = True
    config.USE_NUCLEAR_POINT_SOURCE = False
    config.NUCLEAR_POINT_RA = float(nuc_ra)
    config.NUCLEAR_POINT_DEC = float(nuc_dec)
    config.NUCLEAR_POINT_NONNEGATIVE = True
    config.MAX_SCENE_PIXELS = 1000

    cutouts = []
    try:
        bcd_wcs_list = [
            _rot_wcs(
                n_pix,
                ra0,
                dec0,
                theta_deg=17.0 + 2.0 * i,
                pixel_scale_arcsec=config.PIXEL_SCALE,
            )
            for i in range(n_bcd)
        ]
        scene_w_north, scene_shape, scene_scale_arcsec = _scene_wcs_budgeted(
            bcd_wcs_list,
            (n_pix, n_pix),
            min_scene_pixel_scale_arcsec=(config.PIXEL_SCALE / config.SUPERSAMPLE_FACTOR),
            sky_points_deg=[(ra0, dec0), (nuc_ra, nuc_dec)],
            max_scene_pixels=13000,
            pad_scene_px=int(getattr(config, "NATIVE_SCENE_PAD_PX", 6)),
        )
        n_scene = int(scene_shape[0] * scene_shape[1])
        cx_g, cy_g = scene_w_north.world_to_pixel_values(float(nuc_ra), float(nuc_dec))
        scene_truth = _synthetic_galaxy_scene(scene_shape, float(cx_g), float(cy_g))
        host_truth_scene = solver.host_core_gaussian_column(
            scene_w_north,
            float(nuc_ra),
            float(nuc_dec),
            1.7,
            scene_shape,
        ).reshape(scene_shape) * 1.5e-5
        nps_truth_scene = np.zeros(scene_shape, dtype=float)
        nx_scene, ny_scene = scene_w_north.world_to_pixel_values(float(nuc_ra), float(nuc_dec))
        solver._add_delta_to_image(nps_truth_scene, float(nx_scene), float(ny_scene), 1.3e-5)
        intrinsic_truth = scene_truth + host_truth_scene + nps_truth_scene

        for i in range(n_bcd):
            w_i = bcd_wcs_list[i]
            bg_i = 1.2e-5 + 6.0e-7 * i
            noise_sig = 1.2e-6

            conv_truth = solver._apply_frame_forward_operator(
                intrinsic_truth,
                scene_w_north,
                w_i,
                scene_shape,
                (n_pix, n_pix),
                "ch2",
                is_full_array=True,
            )
            d = (conv_truth + bg_i + rng.normal(0.0, noise_sig, (n_pix, n_pix))).astype(np.float64)
            s = np.full_like(d, noise_sig)
            cutouts.append({
                "data": d,
                "sigma": s,
                "wcs": w_i,
                "raw_wcs": w_i,
                "is_full_array": True,
                "mjd": 58000.0 + i,
                "filename": f"synthetic_native_ch2_{i:03d}_cbcd.fits",
                "epoch_id": i,
                "is_template": True,
            })

            # Cutout must be centered on transient (native pixels) to within 1 px.
            tx, ty = w_i.world_to_pixel_values(ra0, dec0)
            cx = float(w_i.wcs.crpix[0]) - 1.0
            cy = float(w_i.wcs.crpix[1]) - 1.0
            assert abs(float(tx) - cx) <= 1.0
            assert abs(float(ty) - cy) <= 1.0

        # Use a shorter GP scale in this diagnostic so GP can absorb compact
        # central structure when host Gaussian / nuclear point terms are disabled.
        ell, var = 1.8, 1e-7
        results = solver.run_gls_solve(
            cutouts,
            [],
            [],
            {"ell": ell, "var": var},
            (ell, var),
            np.zeros(scene_shape),
            scene_w_north,
            n_bcd,
        )
        assert results is not None
        assert results["scene_shape"] == scene_shape
        assert float(scene_scale_arcsec) <= float(config.PIXEL_SCALE)
        assert np.isfinite(np.asarray(results["model_scene"])).all()
        assert len(results["bcd_backgrounds"]) == n_bcd
        assert np.allclose(np.asarray(results["transient_fluxes"], dtype=float), 0.0, atol=1e-15)

        # Nucleus sky position → scene pixels; Gaussian peak and delta should match within a few pixels.
        scene_wcs = results["scene_wcs"]
        nx, ny = scene_wcs.world_to_pixel_values(float(nuc_ra), float(nuc_dec))
        host_unit = solver.host_core_gaussian_column(
            scene_wcs,
            float(nuc_ra),
            float(nuc_dec),
            float(getattr(config, "HOST_CORE_SIGMA_PX", 1.7)),
            results["scene_shape"],
        ).reshape(results["scene_shape"])
        y_h, x_h = np.unravel_index(int(np.argmax(host_unit)), host_unit.shape)
        assert abs(float(x_h) - float(nx)) <= 5.0
        assert abs(float(y_h) - float(ny)) <= 5.0
        nps_unit = np.zeros(results["scene_shape"], dtype=float)
        solver._add_delta_to_image(nps_unit, float(nx), float(ny), 1.0)
        y_p, x_p = np.unravel_index(int(np.argmax(nps_unit)), nps_unit.shape)
        assert abs(float(x_p) - float(nx)) <= 5.0
        assert abs(float(y_p) - float(ny)) <= 5.0

        out_pdf = _write_native_fit_pdf(n_bcd, cutouts, results)
        assert os.path.exists(out_pdf)
        assert os.path.getsize(out_pdf) > 0
    finally:
        (
            config.USE_HOST_GAUSSIAN_CORE,
            config.HOST_CORE_RA,
            config.HOST_CORE_DEC,
            config.HOST_CORE_SIGMA_PX,
            config.HOST_CORE_NONNEGATIVE,
        ) = prev_host
        (
            config.USE_NUCLEAR_POINT_SOURCE,
            config.NUCLEAR_POINT_RA,
            config.NUCLEAR_POINT_DEC,
            config.NUCLEAR_POINT_NONNEGATIVE,
        ) = prev_nps
        config.MAX_SCENE_PIXELS = prev_max_scene
        config.TRANSIENT_RA, config.TRANSIENT_DEC = prev_transient
