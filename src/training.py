import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from pathlib import Path
import os
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split
import time
import copy


class FactoryDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        """
        Dataset for factory images with multiple channels
        
        Args:
            data_dir (str): Directory containing the numpy arrays
            transform (callable, optional): Optional transform to apply to the data
        """
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.file_list = [f for f in os.listdir(data_dir) if f.endswith('.npy')]
        
        # Check if we have recipe IDs
        self.recipe_ids_path = self.data_dir / 'recipe_ids.npy'
        self.has_recipes = self.recipe_ids_path.exists()
        if self.has_recipes:
            self.recipe_ids = np.load(self.recipe_ids_path)
    
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        # Load numpy array (H×W×C)
        file_path = self.data_dir / self.file_list[idx]
        data_array = np.load(file_path)
        
        # Extract channels
        opacity_channels = data_array[:, :, :4]  # First 4 channels
        recipe_channel = data_array[:, :, 4:5]   # Recipe channel
        direction_channel = data_array[:, :, 5:6]  # Directionality channel
        power_channel = data_array[:, :, 6:7]    # Power coverage channel
        source_sink_channel = data_array[:, :, 7:8]  # Source/sink channel
        
        # Combine all channels and convert to tensor
        # Rearrange from HWC to CHW format for PyTorch
        all_channels = np.concatenate([
            opacity_channels,
            recipe_channel,
            direction_channel, 
            power_channel,
            source_sink_channel
        ], axis=2)
        
        # Convert to PyTorch tensor and rearrange dimensions
        tensor_data = torch.from_numpy(all_channels).permute(2, 0, 1).float()
        
        # Get recipe ID if available
        recipe_id = None
        if self.has_recipes:
            recipe_id = torch.tensor(self.recipe_ids[idx], dtype=torch.long)
        
        # Apply transforms if any
        if self.transform:
            tensor_data = self.transform(tensor_data)
            
        return tensor_data, recipe_id
    


class FactoryTrainer:
    def __init__(self, model, criterion=None, optimizer=None, scheduler=None, 
                 device=None, use_recipe_id=True):
        """
        Trainer for the FactoryAutoencoder
        
        Args:
            model: The FactoryAutoencoder model
            criterion: Loss function (default: MSELoss)
            optimizer: Optimizer (default: Adam)
            scheduler: Learning rate scheduler (optional)
            device: Device to run on (default: cuda if available, else cpu)
            use_recipe_id: Whether to use recipe IDs in training
        """
        self.model = model
        
        # Default criterion if none provided
        self.criterion = criterion if criterion else nn.MSELoss()
        
        # Default optimizer if none provided
        if optimizer is None:
            self.optimizer = optim.Adam(model.parameters(), lr=1e-3)
        else:
            self.optimizer = optimizer
            
        self.scheduler = scheduler
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_recipe_id = use_recipe_id
        
        # Move model to device
        self.model.to(self.device)
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'epochs': 0,
            'best_model_wts': copy.deepcopy(model.state_dict()),
            'best_loss': float('inf')
        }
    
    def train_epoch(self, dataloader):
        """Run one epoch of training"""
        self.model.train()
        running_loss = 0.0
        
        for inputs, recipe_ids in dataloader:
            inputs = inputs.to(self.device)
            
            if recipe_ids is not None and self.use_recipe_id:
                recipe_ids = recipe_ids.to(self.device)
            else:
                recipe_ids = None
                
            # Zero the gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(inputs, recipe_ids)
            
            # Calculate loss
            loss = self.criterion(outputs, inputs)
            
            # Backward pass and optimize
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            
        epoch_loss = running_loss / len(dataloader.dataset)
        return epoch_loss
    
    def validate(self, dataloader):
        """Validate the model"""
        self.model.eval()
        running_loss = 0.0
        
        with torch.no_grad():
            for inputs, recipe_ids in dataloader:
                inputs = inputs.to(self.device)
                
                if recipe_ids is not None and self.use_recipe_id:
                    recipe_ids = recipe_ids.to(self.device)
                else:
                    recipe_ids = None
                
                # Forward pass
                outputs = self.model(inputs, recipe_ids)
                
                # Calculate loss
                loss = self.criterion(outputs, inputs)
                
                running_loss += loss.item() * inputs.size(0)
                
        epoch_loss = running_loss / len(dataloader.dataset)
        return epoch_loss
        
    def train(self, dataset, batch_size=32, epochs=50, validation_split=0.2, 
              early_stopping_patience=10, save_path=None):
        """
        Train the model
        
        Args:
            dataset: FactoryDataset instance
            batch_size: Batch size for training
            epochs: Number of epochs to train
            validation_split: Fraction of data to use for validation
            early_stopping_patience: Number of epochs with no improvement before stopping
            save_path: Path to save the best model weights
            
        Returns:
            Training history
        """
        # Split dataset into train and validation
        val_size = int(validation_split * len(dataset))
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        # Variables for early stopping
        no_improve_epochs = 0
        
        print(f"Training on {self.device}")
        print(f"Starting training for {epochs} epochs...")
        
        start_time = time.time()
        
        for epoch in range(epochs):
            # Train one epoch
            train_loss = self.train_epoch(train_loader)
            
            # Validate
            val_loss = self.validate(val_loader)
            
            # Update learning rate if scheduler is provided
            if self.scheduler:
                self.scheduler.step(val_loss)
            
            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['epochs'] += 1
            
            # Print progress
            print(f"Epoch {epoch+1}/{epochs} - "
                  f"Train Loss: {train_loss:.6f} - "
                  f"Val Loss: {val_loss:.6f}")
            
            # Check if this is the best model
            if val_loss < self.history['best_loss']:
                print(f"Validation loss improved from {self.history['best_loss']:.6f} to {val_loss:.6f}")
                self.history['best_loss'] = val_loss
                self.history['best_model_wts'] = copy.deepcopy(self.model.state_dict())
                
                # Save model if path is provided
                if save_path:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'loss': val_loss,
                    }, save_path)
                    print(f"Model saved to {save_path}")
                
                no_improve_epochs = 0
            else:
                no_improve_epochs += 1
                print(f"No improvement for {no_improve_epochs} epochs")
            
            # Early stopping
            if no_improve_epochs >= early_stopping_patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
        
        # Calculate training time
        time_elapsed = time.time() - start_time
        print(f"Training completed in {time_elapsed//60:.0f}m {time_elapsed%60:.0f}s")
        
        # Load best model weights
        self.model.load_state_dict(self.history['best_model_wts'])
        
        return self.history
    
    def visualize_results(self, dataloader, num_samples=5):
        """Visualize reconstruction results"""
        self.model.eval()
        
        # Get a batch of samples
        inputs, recipe_ids = next(iter(dataloader))
        inputs = inputs[:num_samples].to(self.device)
        
        if recipe_ids is not None and self.use_recipe_id:
            recipe_ids = recipe_ids[:num_samples].to(self.device)
        else:
            recipe_ids = None
        
        # Get reconstructions
        with torch.no_grad():
            outputs = self.model(inputs, recipe_ids)
        
        # Move tensors to CPU for visualization
        inputs = inputs.cpu().numpy()
        outputs = outputs.cpu().numpy()
        
        # Plot results
        fig, axes = plt.subplots(num_samples, 2, figsize=(10, num_samples*3))
        
        for i in range(num_samples):
            # For visualization, we'll use the first 3 opacity channels as RGB
            # and the 4th as alpha if applicable
            input_img = np.transpose(inputs[i, :3], (1, 2, 0))
            output_img = np.transpose(outputs[i, :3], (1, 2, 0))
            
            # Normalize to [0, 1] for display
            input_img = (input_img - input_img.min()) / (input_img.max() - input_img.min() + 1e-8)
            output_img = (output_img - output_img.min()) / (output_img.max() - output_img.min() + 1e-8)
            
            # Plot
            axes[i, 0].imshow(input_img)
            axes[i, 0].set_title("Input")
            axes[i, 0].axis('off')
            
            axes[i, 1].imshow(output_img)
            axes[i, 1].set_title("Reconstruction")
            axes[i, 1].axis('off')
        
        plt.tight_layout()
        return fig