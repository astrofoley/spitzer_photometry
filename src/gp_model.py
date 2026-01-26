"""src/gp_model.py"""
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform
from . import config

def matern32_kernel(coords, length_scale, variance):
    dists = squareform(pdist(coords, metric='euclidean'))
    sqrt3_d = np.sqrt(3) * dists / length_scale
    K = variance * (1 + sqrt3_d) * np.exp(-sqrt3_d)
    return K

def optimize_hyperparameters(template_data_list):
    print("Optimizing GP Hyperparameters...")
    if not template_data_list:
        return config.INIT_LENGTH_SCALE, config.INIT_VARIANCE

    # Use first template stamp
    # Safely handle potential NaNs
    data = np.nan_to_num(template_data_list[0]['data'])
    sigma = np.nan_to_num(template_data_list[0]['sigma'])
    
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
            K = matern32_kernel(coords, ell, var)
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
