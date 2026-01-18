from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np

def reconstruct_curves(Z, n_comp, pca, scaler):
    """
    Reconstruct an array of curves using only the first n_components from a fitted PCA model.
    
    Parameters:
    -----------
    Z : ndarray, shape(n_samples, n_comp_max)
        Components of the curves to reconstruct
    n_comp : int
        Number of PCA components to use for reconstruction
    pca : fitted PCA object
    scaler : fitted StandardScaler object
        
    Returns:
    --------
    curves_reconstructed : ndarray, shape(n_samples, n_features)
        Reconstructed curves
    """
    assert n_comp<=Z.shape[1], "Not enough components in pca"
    # Reconstruct using the first n_comp components
    Z_reconstructed = np.dot(Z[:, :n_comp], pca.components_[:n_comp, :])
    # Add back the mean (PCA centers the data) and inverse scaling 
    curves_reconstructed = scaler.inverse_transform(Z_reconstructed + pca.mean_)
    return curves_reconstructed

def build_reconstructed_curve(curve, n_comp, pca, scaler):
    """
    build the reconstruction of a single curve using only the first n_components from a fitted PCA model.
    
    Parameters:
    -----------
    curve : array-like, shape (n_features,)
        Single data point to transform and reconstruct
    n_comp : int
        Number of PCA components to use for transformation and reconstruction
    pca : fitted PCA object
    scaler : fitted StandardScaler object
        
    Returns:
    --------
    reconstructed : ndarray, shape (n_features,)
        Reconstructed curve
    """
    # perform scaling and pca.mean_
    curve = scaler.transform([curve])[0] - pca.mean_
    z     = pca.transform([curve])[0] # there is only one curve
    # Reconstruct using only first n_comp
    curve_rec = np.dot(z[:n_comp], pca.components_[:n_comp, :])
    # Add back the mean and inverse scaling
    return scaler.inverse_transform([curve_rec + pca.mean_])[0]

def compute_pca(curves, n_comp_max, scale_data=True):
    """
    Perform PCA
    
    Parameters:
    -----------
    curves : ndarray, shape (n_samples, n_features)
            Input data (Gaussian curves)
    n_comp_max : int
        Maximum number of components
    scale_data : bool, default=True
        Whether to scale (standardize) the data before PCA
    
    Returns:
    Z : ndarray, shape(n_samples, n_comp_max)
        principal components
        Cumulative explained variance ratio
    pca : PCA object
        Fitted PCA model
    scaler : StandardScaler object
        Fitted standard scaler
    """    
# Scale data as requested
    scaler = StandardScaler(with_std=scale_data)
    curves_scaled = scaler.fit_transform(curves)
    
    # Fit PCA with maximum components. Z is an ndarray of size (n,n_comp_max)
    pca = PCA(n_components=n_comp_max)
    Z   = pca.fit_transform(curves_scaled)  # Fit and transform
    
    return Z, pca, scaler

def analyze_pca(curves, n_comp_max, scale_data=True):
    """
    Perform PCA and compute MSE reconstruction error for n_comp in range(1, n_comp_max + 1)

    Parameters:
    -----------
    curves : ndarray, shape (n_samples, n_features)
        Input data (Gaussian curves)
    n_comp_max : int
        Maximum number of components to test
    scale_data : bool, default=True
        Whether to scale (standardize) the data before PCA

    Returns:
    --------
    reconstruction_errors : ndarray
        MSE for each components
    explained_variance : ndarray
        Cumulative explained variance ratio
    pca_model : PCA object
        Fitted PCA model
    scaler : StandardScaler object
    """
    
    # Fit PCA with maximum components
    Z, pca, scaler = compute_pca(curves, n_comp_max, scale_data)  # Fit and transform
    
    # Calculate reconstruction error by increasing the number of components
    reconstruction_errors = []
    for n_comp in range(1, n_comp_max + 1):
        # compute the curves reconstructed when using n_comp components
        curves_reconstructed  = reconstruct_curves(Z, n_comp, pca, scaler)
        # Calculate MSE between original and reconstructed
        mse = np.mean((curves - curves_reconstructed) ** 2)
        reconstruction_errors.append(mse)
    
    return Z, np.array(reconstruction_errors), pca.explained_variance_ratio_, pca, scaler

def fit_gmm(Z, n_components=10, random_state=None):
    """
    Fit a Gaussian Mixture Model on the latent space Z.
    
    Parameters:
    -----------
    Z : ndarray
        Latent vectors (PCA components)
    n_components : int
        Number of mixture components
    random_state : int, optional
    
    Returns:
    --------
    gmm : GaussianMixture object
        Fitted GMM model
    """
    from sklearn.mixture import GaussianMixture
    gmm = GaussianMixture(n_components=n_components, covariance_type='full', random_state=random_state)
    gmm.fit(Z)
    return gmm

def generate_new_curves_gmm(gmm, pca, scaler, n_samples=1):
    """
    Sample new points from GMM and reconstruct them to the original space.
    
    Parameters:
    -----------
    gmm : GaussianMixture object
        Fitted GMM model
    pca : PCA object
        Fitted PCA model
    scaler : StandardScaler object
        Fitted scaler
    n_samples : int
        Number of curves to generate
        
    Returns:
    --------
    curves_new : ndarray
        Generated curves
    z_new : ndarray
        Latent points sampled
    """
    # Sample from GMM
    z_new, _ = gmm.sample(n_samples)
    
    # Reconstruct (Inverse PCA + Inverse Scaling)
    # Note: reconstruct_curves function might need the original Z shape or logic
    # But usually: X_rec = (Z @ components) + mean 
    # Let's do it manually to be safe and efficient or use pca.inverse_transform if available
    
    curves_scaled = pca.inverse_transform(z_new)
    curves_new = scaler.inverse_transform(curves_scaled)
    
    return curves_new, z_new