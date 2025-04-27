import numpy as np
from src.representation import center_in_N

import torch
import torch.nn as nn
import torch.nn.functional as F






class FactoryAutoencoder(nn.Module):
    def __init__(self, c: int, h: int, w: int,
                 hidden_dim=64, output_channels=1,
                 max_indices=100, embedding_dim=32):
        super(FactoryAutoencoder, self).__init__()
        
        self.use_index_embedding = embedding_dim > 0
        self.h, self.w = h, w
        
        # Encoder with dimensionality reduction
        self.encoder = nn.Sequential(
            nn.Conv2d(c, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            nn.MaxPool2d(2),  # Reduce spatial dimensions
            nn.Conv2d(hidden_dim, hidden_dim*2, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim*2),
            nn.ReLU(),
        )
        
        # Store dimensions after pooling
        self.encoded_h, self.encoded_w = h // 2, w // 2
        
        if self.use_index_embedding:
            # Index embedding with configurable dimensions
            self.index_embedding = nn.Embedding(num_embeddings=max_indices, embedding_dim=embedding_dim)
            self.index_projection = nn.Linear(embedding_dim, hidden_dim*2)
            
            # Spatial attention mechanism
            self.spatial_attention = nn.Sequential(
                nn.Conv2d(hidden_dim*2, hidden_dim, kernel_size=1),
                nn.ReLU(),
                nn.Conv2d(hidden_dim, 1, kernel_size=1),
                nn.Sigmoid()
            )
        
        # Decoder with upsampling to match encoder's reduction
        self.decoder = nn.Sequential(
            nn.Conv2d(hidden_dim*2, hidden_dim*2, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim*2),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),  # Match encoder's pooling
            nn.Conv2d(hidden_dim*2, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, output_channels, kernel_size=1)
        )
    
    def get_embedding(self, x, index=None):
        """Get embeddings suitable for Siamese networks"""
        encoded = self.encoder(x)
        
        if self.use_index_embedding and index is not None:
            idx_emb = self.index_embedding(index)
            idx_proj = self.index_projection(idx_emb)
            b = x.size(0)
            idx_proj = idx_proj.view(b, -1, 1, 1).expand(-1, -1, self.encoded_h, self.encoded_w)
            
            attention = self.spatial_attention(encoded)
            encoded = encoded * attention + idx_proj * (1 - attention)
            
        # Global pooling for fixed-size representation
        global_features = F.adaptive_max_pool2d(encoded, 1).squeeze(-1).squeeze(-1)
        return global_features
    
    def get_normalized_embedding(self, x, index=None):
        """Get L2-normalized embeddings for cosine similarity"""
        emb = self.get_embedding(x, index)
        return F.normalize(emb, p=2, dim=1)
    
    def forward(self, x, index=None):
        """Full autoencoder forward pass"""
        encoded = self.encoder(x)
        
        if self.use_index_embedding and index is not None:
            # Process index embedding
            idx_emb = self.index_embedding(index)
            idx_proj = self.index_projection(idx_emb)
            b = x.size(0)
            idx_proj = idx_proj.view(b, -1, 1, 1).expand(-1, -1, self.encoded_h, self.encoded_w)
            
            # Apply attention mechanism
            attention = self.spatial_attention(encoded)
            encoded = encoded * attention + idx_proj * (1 - attention)
        
        # Decode to get output
        decoded = self.decoder(encoded)
        return decoded
    

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
