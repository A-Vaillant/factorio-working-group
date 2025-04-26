import numpy as np
from src.representation import center_in_N

import torch
import torch.nn as nn
import torch.nn.functional as F


# Example usage
def test_centering():
    # Test with different sized matrices
    matrices = [
        np.ones((3, 3)),                # 3x3 matrix
        np.ones((5, 7)),                # 5x7 matrix
        np.ones((10, 8)),               # 10x8 matrix
        np.ones((3, 3, 4))              # 3x3 matrix with 4 channels
    ]
    
    for i, mat in enumerate(matrices):
        centered = center_in_N(mat, N=15)
        print(f"Original shape: {mat.shape}, Centered shape: {centered.shape}")
        
        # Visualize the result for 2D matrices
        if mat.ndim == 2 or (mat.ndim == 3 and mat.shape[2] == 1):
            print(f"Center positions where matrix was placed:")
            binary_mask = np.where(centered > 0, 1, 0)
            if binary_mask.ndim == 3:
                binary_mask = binary_mask[:,:,0]
            print(binary_mask.astype(int))
        print("-" * 30)


class FactoryAutoencoder(nn.Module):
    def __init__(self, c: int, h: int, w: int,
                 hidden_dim=64, output_channels=1):
        super(FactoryAutoencoder, self).__init__()
        
        self.input_size = w  # Assuming that w = h.

        # Encoder for processing the N-channel input
        self.encoder = nn.Sequential(
            nn.Conv2d(c, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim*2, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim*2),
            nn.ReLU(),
        )
        
        # Index embedding
        self.index_embedding = nn.Embedding(num_embeddings=100, embedding_dim=32)
        self.index_projection = nn.Linear(32, hidden_dim*2)
        
        # Spatial attention mechanism to focus on relevant positions
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(hidden_dim*2, hidden_dim, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Decoder for generating the output matrix
        self.decoder = nn.Sequential(
            nn.Conv2d(hidden_dim*2, hidden_dim*2, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim*2),
            nn.ReLU(),
            nn.Conv2d(hidden_dim*2, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, output_channels, kernel_size=1)
        )
    
    def forward(self, x, index):
        batch_size, _, H, W = x.shape
        
        # Process the N-channel matrix
        features = self.encoder(x)
        
        # Process the index and broadcast to feature map dimensions
        index_embedding = self.index_embedding(index)  # [B, 32]
        index_features = self.index_projection(index_embedding)  # [B, hidden_dim*2]
        index_features = index_features.view(batch_size, -1, 1, 1).expand(-1, -1, H, W)
        
        # Combine features with index information using element-wise multiplication
        combined_features = features * index_features
        
        # Compute spatial attention map
        attention_map = self.spatial_attention(combined_features)
        attended_features = combined_features * attention_map
        
        # Generate output matrix
        output_matrix = self.decoder(attended_features)
        
        return output_matrix, attention_map


def train_and_test_model():
    # Initialize model
    model = FactoryAutoencoder()
    
    # Sample input
    batch_size = 8
    H, W = 64, 64
    x = torch.randn(batch_size, 4, H, W)
    index = torch.randint(0, 100, (batch_size,))
    
    # Forward pass
    output_matrix, position, attention_map = model(x, index)
    
    print(f"Input shape: {x.shape}")
    print(f"Output matrix shape: {output_matrix.shape}")
    print(f"Predicted position shape: {position.shape}")
    print(f"Attention map shape: {attention_map.shape}")
    
    return model

if __name__ == "__main__":
    model = train_and_test_model()
