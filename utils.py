import numpy as np
import matplotlib.pyplot as plt

def plot_reconstruction_comparison(x, original, reconstructed, title="Reconstruction Comparison", num_samples=3):
    """
    Plot original vs reconstructed curves.
    """
    indices = np.random.choice(len(original), num_samples, replace=False)
    
    fig, axes = plt.subplots(1, num_samples, figsize=(5*num_samples, 4))
    if num_samples == 1:
        axes = [axes]
        
    for i, ax in enumerate(axes):
        idx = indices[i]
        ax.plot(x, original[idx], 'k-', label='Original', linewidth=2)
        ax.plot(x, reconstructed[idx], 'r--', label='Reconstructed', linewidth=2)
        ax.set_title(f"Sample {idx}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

def plot_latent_space(Z, labels=None, title="Latent Space"):
    """
    Plot first 2 dimensions of latent space.
    """
    plt.figure(figsize=(8, 6))
    if labels is not None:
        scatter = plt.scatter(Z[:, 0], Z[:, 1], c=labels, cmap='viridis', alpha=0.6, s=10)
        plt.colorbar(scatter)
    else:
        plt.scatter(Z[:, 0], Z[:, 1], alpha=0.6, s=10)
        
    plt.xlabel("Latent Dim 1")
    plt.ylabel("Latent Dim 2")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.show()

def plot_generated_curves(x, generated_curves, title="Generated Curves", num_samples=10):
    """
    Plot a few generated curves.
    """
    x = np.asarray(x).flatten()
    generated_curves = np.asarray(generated_curves)
    
    plt.figure(figsize=(10, 5))
    for i in range(min(num_samples, len(generated_curves))):
        plt.plot(x, generated_curves[i], alpha=0.5)
        
    plt.title(title)
    plt.xlabel("x")
    plt.grid(True, alpha=0.3)
    plt.show()

def plot_curves_from_clusters(x, curves, labels, n_clusters_show=4, n_samples_per_cluster=3):
    """
    Plot samples of curves generated from specific clusters.
    """
    x = np.asarray(x).flatten()
    curves = np.asarray(curves)
    labels = np.asarray(labels).flatten()
    
    unique_labels = np.unique(labels)
    n_display = min(len(unique_labels), n_clusters_show)
    
    fig, axes = plt.subplots(1, n_display, figsize=(4*n_display, 3), sharey=True)
    if n_display == 1: axes = [axes]
    
    for i in range(n_display):
        cluster_id = unique_labels[i]
        cluster_curves = curves[labels == cluster_id]
        
        ax = axes[i]
        # Plot a few random samples from this cluster
        indices = np.random.choice(len(cluster_curves), min(len(cluster_curves), n_samples_per_cluster), replace=False)
        for idx in indices:
            ax.plot(x, cluster_curves[idx], alpha=0.7)
            
        ax.set_title(f"Cluster {cluster_id}")
        ax.grid(True, alpha=0.3)
        if i == 0: ax.set_ylabel("Amplitude")
        ax.set_xlabel("x")
        
    plt.suptitle("Exemples de courbes par cluster")
    plt.tight_layout()
    plt.show()

