import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

class Autoencoder(nn.Module):
    """
    A generic Autoencoder with MLP architecture.
    """
    def __init__(self, input_dim, latent_dim, hidden_dims=[128, 64]):
        super(Autoencoder, self).__init__()
        
        # Encoder
        encoder_layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            encoder_layers.append(nn.Linear(in_dim, h_dim))
            encoder_layers.append(nn.ReLU())
            in_dim = h_dim
        encoder_layers.append(nn.Linear(in_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Decoder
        decoder_layers = []
        in_dim = latent_dim
        hidden_dims_rev = hidden_dims[::-1]
        for h_dim in hidden_dims_rev:
            decoder_layers.append(nn.Linear(in_dim, h_dim))
            decoder_layers.append(nn.ReLU())
            in_dim = h_dim
        decoder_layers.append(nn.Linear(in_dim, input_dim))
        # Note: We do not enforce positivity at the output (e.g. Softplus) 
        # to keep the architecture standard, but it could be added if needed.
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z
        
    def encode(self, x):
        return self.encoder(x)
        
    def decode(self, z):
        return self.decoder(z)

def train_autoencoder(model, data, epochs=50, batch_size=32, lr=1e-3, device='cpu'):
    """
    Train the autoencoder.
    
    Parameters:
    -----------
    model : Autoencoder instance
    data : ndarray (n_samples, n_features)
    epochs : int
    batch_size : int
    lr : float
    device : str ('cpu' or 'cuda')
    
    Returns:
    --------
    losses : list of float
    """
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    # Prepare data
    tensor_data = torch.FloatTensor(data)
    dataset = TensorDataset(tensor_data)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model.train()
    losses = []
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch in loader:
            x = batch[0].to(device)
            
            optimizer.zero_grad()
            x_hat, _ = model(x)
            loss = criterion(x_hat, x)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * x.size(0)
            
        avg_loss = epoch_loss / len(data)
        losses.append(avg_loss)
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.6f}")
            
    return losses

def get_latent_space(model, data, device='cpu'):
    """
    Project data to latent space.
    """
    model.eval()
    with torch.no_grad():
        x = torch.FloatTensor(data).to(device)
        z = model.encode(x)
    return z.cpu().numpy()

def fit_gmm_on_latent(Z, n_components=10, random_state=None):
    """
    Fit GMM on latent vectors Z.
    """
    gmm = GaussianMixture(n_components=n_components, covariance_type='full', random_state=random_state)
    gmm.fit(Z)
    return gmm

def generate_new_curves_ae(gmm, model, scaler=None, n_samples=1, device='cpu'):
    """
    Generate new curves using GMM sampling + Decoder.
    If a scaler was used on data before training AE, provide it to inverse transform.
    """
    # 1. Sample from GMM
    z_new_np, _ = gmm.sample(n_samples)
    z_new = torch.FloatTensor(z_new_np).to(device)
    
    # 2. Decode
    model.eval()
    with torch.no_grad():
        x_rec = model.decode(z_new).cpu().numpy()
        
    # 3. Inverse scale if necessary
    if scaler is not None:
        x_rec = scaler.inverse_transform(x_rec)
        
    return x_rec, z_new_np
