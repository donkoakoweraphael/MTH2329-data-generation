import numpy as np

def compute_mse(original_curves, reconstructed_curves):
    """
    Compute Mean Squared Error (MSE) between original and reconstructed curves.
    """
    return np.mean((original_curves - reconstructed_curves) ** 2)

def check_negative_values(curves):
    """
    Check for negative values in the curves.
    Returns the percentage of negative values and the number of curves containing at least one negative value.
    """
    total_points = curves.size
    negative_points = np.sum(curves < 0)
    percent_negative = (negative_points / total_points) * 100
    
    curves_with_neg = np.sum(np.any(curves < 0, axis=1))
    
    return percent_negative, curves_with_neg

def estimate_gaussian_params(x, y):
    """
    Estimate mu and sigma from a discretized Gaussian curve y = f(x).
    Using weighted average:
    mu = sum(x * y) / sum(y)
    sigma^2 = sum((x - mu)^2 * y) / sum(y)
    
    Parameters:
    -----------
    x : ndarray (nbx,)
        x coordinates
    y : ndarray (nbx,) or (N, nbx)
        y values (curve)
    
    Returns:
    --------
    mu_est : float or ndarray
    sigma_est : float or ndarray
    """
    # Handle single curve
    if y.ndim == 1:
        y = y.reshape(1, -1)
        
    # Ensure y is positive for stability (though original should be)
    # y = np.maximum(y, 1e-10) 
    
    sum_y = np.sum(y, axis=1)
    # Avoid division by zero
    sum_y[sum_y == 0] = 1.0
    
    mu_est = np.sum(x * y, axis=1) / sum_y
    
    # Variance
    # (x - mu)^2. Need broadcasting.
    # x shape (nbx,), mu shape (N,)
    # We can do: Var = E[x^2] - (E[x])^2
    
    mu_est_reshaped = mu_est.reshape(-1, 1) # (N, 1)
    
    # Efficient calculation: sum(y * (x - mu)^2)
    # We'll use the loop or broadcasting
    # (x - mu)^2 is (N, nbx)
    
    # x is (nbx,) -> (1, nbx)
    x_grid = x.reshape(1, -1)
    
    var_est = np.sum(y * (x_grid - mu_est_reshaped)**2, axis=1) / sum_y
    sigma_est = np.sqrt(var_est)
    
    if y.shape[0] == 1:
        return mu_est[0], sigma_est[0]
        
    return mu_est, sigma_est

def evaluate_generation_quality(original_params, generated_curves, x_grid):
    """
    Compare statistics of generated curves with original parameters.
    
    Parameters:
    -----------
    original_params : tuple (mu_orig, sigma_orig)
        Arrays of original parameters used to generate the dataset.
        OR None if we want to compare to the estimated params of the original dataset.
    generated_curves : ndarray
    x_grid : ndarray
    
    Returns:
    --------
    dict with metrics
    """
    mu_gen, sigma_gen = estimate_gaussian_params(x_grid, generated_curves)
    
    # If we had the parameters of the specific curves that were reconstructed, we could compare element-wise.
    # But for generation (sampling), we compare distributions of parameters.
    
    # Statistics of the parameters
    results = {
        'mu_mean_gen': np.mean(mu_gen),
        'mu_std_gen': np.std(mu_gen),
        'sigma_mean_gen': np.mean(sigma_gen),
        'sigma_std_gen': np.std(sigma_gen)
    }
    
    if original_params is not None:
        mu_orig, sigma_orig = original_params
        results['mu_mean_orig'] = np.mean(mu_orig)
        results['mu_std_orig'] = np.std(mu_orig)
        results['sigma_mean_orig'] = np.mean(sigma_orig)
        results['sigma_std_orig'] = np.std(sigma_orig)
        
        # Divergence between distributions of mu (e.g. difference of means)
        results['diff_mu_mean'] = np.abs(results['mu_mean_gen'] - results['mu_mean_orig'])
        results['diff_sigma_mean'] = np.abs(results['sigma_mean_gen'] - results['sigma_mean_orig'])
        
    return results, (mu_gen, sigma_gen)
