"""src/gp_model.py"""
import warnings
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform
from . import config


def normalize_matern_order(name) -> str:
    """Return 'matern12' or 'matern32' for scene-prior / plotting."""
    s = str(name or "matern32").strip().lower().replace(" ", "")
    if s in ("matern12", "m12", "12", "exp", "exponential", "matern_12", "matern1/2", "1/2"):
        return "matern12"
    return "matern32"


def matern12_kernel(coords, length_scale, variance):
    """Matérn nu=1/2 (exponential) covariance on Euclidean distances."""
    dists = squareform(pdist(coords, metric="euclidean"))
    d = dists / max(float(length_scale), 1e-30)
    return float(variance) * np.exp(-d)


def matern32_kernel(coords, length_scale, variance):
    dists = squareform(pdist(coords, metric='euclidean'))
    sqrt3_d = np.sqrt(3) * dists / length_scale
    K = variance * (1 + sqrt3_d) * np.exp(-sqrt3_d)
    return K


def scene_kernel_matrix(coords, length_scale, variance, order=None):
    """Stationary isotropic Matérn kernel on grid coordinates (scene pixel units)."""
    ord_ = normalize_matern_order(order if order is not None else getattr(config, "GP_MATERN_ORDER", "matern32"))
    ls = float(max(float(length_scale), 1e-30))
    if ord_ == "matern12":
        return matern12_kernel(coords, ls, variance)
    return matern32_kernel(coords, ls, variance)


def build_scene_prior_inverse(n_scene, ell, var, scene_shape, ell2=None, var2=None):
    """
    Precision matrix Q^{-1} for the scene GP prior (Matérn 1/2 or 3/2 per config.GP_MATERN_ORDER).
    Falls back to diagonal if n_scene exceeds config.MAX_SCENE_PIXELS.
    """
    use_two_scale = (ell2 is not None) and (var2 is not None)

    if n_scene > config.MAX_SCENE_PIXELS:
        smooth = float(max(0.0, getattr(config, 'GP_FALLBACK_NEIGHBOR_SMOOTHNESS', 0.0)))
        eff_var = float(var)
        if use_two_scale:
            eff_var += float(var2)
        warnings.warn(
            f"Scene size {n_scene} exceeds MAX_SCENE_PIXELS ({config.MAX_SCENE_PIXELS}); "
            f"using {'smoothed ' if smooth > 0 else ''}diagonal GP prior (variance={eff_var}).",
            UserWarning,
            stacklevel=2,
        )
        inv_var = 1.0 / max(float(eff_var), 1e-12)
        Qinv = np.eye(n_scene) * inv_var
        if smooth <= 0.0:
            return Qinv
        h, w = int(scene_shape[0]), int(scene_shape[1])
        if h * w != n_scene:
            return Qinv
        lam = float(smooth) * inv_var
        idx = np.arange(n_scene, dtype=int).reshape(h, w)
        right_i = idx[:, :-1].ravel()
        right_j = idx[:, 1:].ravel()
        down_i = idx[:-1, :].ravel()
        down_j = idx[1:, :].ravel()
        for a, b in ((right_i, right_j), (down_i, down_j)):
            Qinv[a, a] += lam
            Qinv[b, b] += lam
            Qinv[a, b] -= lam
            Qinv[b, a] -= lam
        return Qinv
    y, x = np.mgrid[0:scene_shape[0], 0:scene_shape[1]]
    coords = np.vstack([y.ravel(), x.ravel()]).T
    ell_s = float(ell) * float(config.SUPERSAMPLE_FACTOR)
    Q = scene_kernel_matrix(coords, ell_s, var)
    if use_two_scale:
        Q += scene_kernel_matrix(coords, float(ell2) * float(config.SUPERSAMPLE_FACTOR), float(var2))
    Q += np.eye(n_scene) * 1e-6
    return np.linalg.inv(Q)

def optimize_hyperparameters(template_data_list):
    print("Optimizing GP Hyperparameters...")
    if not template_data_list:
        return config.INIT_LENGTH_SCALE, config.INIT_VARIANCE

    # Use first template stamp. Do not nan_to_num sigma: inf marks masked CR pixels.
    data = np.nan_to_num(template_data_list[0]['data'])
    sigma = np.asarray(template_data_list[0]['sigma'], dtype=np.float64)
    
    # Subsample to avoid slow optimization
    # (Use central 20x20 region)
    h, w = data.shape
    cy, cx = h//2, w//2
    sz = 10
    
    sub_data = data[cy-sz:cy+sz, cx-sz:cx+sz]
    sub_sigma = sigma[cy-sz:cy+sz, cx-sz:cx+sz]
    
    # Skip if empty/masked
    if np.all(sub_data == 0) or np.all(np.isinf(sub_sigma)):
        return config.INIT_LENGTH_SCALE, config.INIT_VARIANCE

    y, x = np.mgrid[0:sub_data.shape[0], 0:sub_data.shape[1]]
    coords = np.vstack([y.ravel(), x.ravel()]).T
    
    flat_data = sub_data.ravel()
    flat_sigma = sub_sigma.ravel()
    
    # Mask infinite sigmas
    valid = np.isfinite(flat_sigma)
    coords = coords[valid]
    flat_data = flat_data[valid]
    flat_sigma = flat_sigma[valid]

    def neg_log_likelihood(params):
        ln_ell, ln_var = params
        ell = np.exp(ln_ell)
        var = np.exp(ln_var)
        
        try:
            K = scene_kernel_matrix(coords, ell, var)
            K += np.diag(flat_sigma**2)
            
            L = np.linalg.cholesky(K)
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, flat_data))
            nll = 0.5 * np.dot(flat_data, alpha) + np.sum(np.log(np.diag(L)))
            return nll
        except:
            return np.inf

    init_params = [np.log(config.INIT_LENGTH_SCALE), np.log(config.INIT_VARIANCE)]
    res = minimize(neg_log_likelihood, init_params, method='L-BFGS-B', bounds=[(-2, 3), (-5, 5)])
    
    best_ell = np.exp(res.x[0])
    best_var = np.exp(res.x[1])
    
    print(f"Optimal Hyperparams: Length Scale={best_ell:.2f}, Variance={best_var:.2f}")
    return best_ell, best_var
