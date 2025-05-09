import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.tensorboard import SummaryWriter
import os
from datetime import datetime
import logging
from src.pipeline.loaders import collate_numpy_matrices
from src.processor import EntityPuncher
from src.pipeline.datasets import load_dataset
# from src.pipeline.filters import (Required,
#                                   RecipeWhitelist,
#                                   SizeRestrictor,
#                                   Blacklist,
#                                   Whitelist)

logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class QCNNTrainer:
    def __init__(
        self, 
        model, 
        learning_rate=0.001,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        log_dir='runs'
    ):
        self.model = model.to(device).float()
        self.device = device
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.base_criterion = nn.MSELoss()
        
        # Set up TensorBoard logging
        current_time = datetime.now().strftime('%b%d_%H-%M-%S')
        self.log_dir = os.path.join(log_dir, current_time)
        self.writer = SummaryWriter(log_dir=self.log_dir)
    
    def calculate_loss(self, outputs, targets):
        """
        Base loss calculation - override this method to implement custom loss terms
        Returns tuple of (total_loss, component_losses_dict)
        """
        base_loss = self.base_criterion(outputs, targets)
        return base_loss, {"base_loss": base_loss.item()}
    
    def train_epoch(self, dataloader, epoch):
        """Train for one epoch"""
        self.model.train()
        running_losses = {}
        
        for i, (Xs, Ys, c) in enumerate(dataloader):
            Xs = Xs.to(self.device)
            Ys = Ys.to(self.device)
            if c is not None:
                c = c.to(self.device)
            
            # Forward pass
            outputs = self.model(Xs, c)
            
            # Calculate loss (using the method that can be overridden)
            loss, loss_components = self.calculate_loss(outputs, Ys)
            
            # Backward and optimize
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Update running losses
            for name, value in loss_components.items():
                running_losses[name] = running_losses.get(name, 0.0) + value
            
            # Log batch metrics (every 10 batches)
            if i % 10 == 0:
                step = epoch * len(dataloader) + i
                for name, value in loss_components.items():
                    self.writer.add_scalar(f'Batch/{name}', value, step)
                
                # Periodically log sample visualizations
                if i % 50 == 0:
                    self._log_matrix_samples(Xs[0], outputs[0], Ys[0], step)
        
        # Compute epoch averages
        avg_losses = {name: value / len(dataloader) for name, value in running_losses.items()}
        return avg_losses
    
    def validate(self, dataloader, epoch):
        """Validate the model"""
        self.model.eval()
        running_losses = {}
        
        with torch.no_grad():
            for Xs, Ys, c in dataloader:
                Xs = Xs.to(self.device)
                Ys = Ys.to(self.device)
                if c is not None:
                    c = c.to(self.device)
                
                # Forward pass
                outputs = self.model(Xs, c)
                
                # Calculate loss
                _, loss_components = self.calculate_loss(outputs, Ys)
                
                # Update running losses
                for name, value in loss_components.items():
                    running_losses[name] = running_losses.get(name, 0.0) + value
        
        # Compute validation averages
        avg_losses = {name: value / len(dataloader) for name, value in running_losses.items()}
        return avg_losses
    
    def _log_matrix_samples(self, input_sample, output_sample, target_sample, step):
        """Log sample visualizations to TensorBoard"""
        # Log input channels
        for c in range(input_sample.shape[0]):
            self.writer.add_image(
                f'Input/Channel_{c}', 
                input_sample[c:c+1], 
                step, 
                dataformats='CHW'
            )
        
        # Log output channels
        for c in range(output_sample.shape[0]):
            self.writer.add_image(
                f'Output/Channel_{c}', 
                output_sample[c:c+1], 
                step, 
                dataformats='CHW'
            )
            
            # Log target channels
            self.writer.add_image(
                f'Target/Channel_{c}', 
                target_sample[c:c+1], 
                step, 
                dataformats='CHW'
            )
            
            # Log difference
            diff = torch.abs(output_sample[c:c+1] - target_sample[c:c+1])
            self.writer.add_image(
                f'Diff/Channel_{c}', 
                diff, 
                step, 
                dataformats='CHW'
            )

    def train(self, train_dataloader, val_dataloader=None, num_epochs=100, save_path=None):
        """Full training loop with validation"""
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            # Train
            train_losses = self.train_epoch(train_dataloader, epoch)
            
            # Log training metrics
            for name, value in train_losses.items():
                self.writer.add_scalar(f'Epoch/Train_{name}', value, epoch)
            
            # Print training stats
            print(f'Epoch {epoch+1}/{num_epochs}')
            train_loss_str = ", ".join([f"{name}: {value:.4f}" for name, value in train_losses.items()])
            print(f'  Train: {train_loss_str}')
            
            # Validate if dataloader provided
            if val_dataloader:
                val_losses = self.validate(val_dataloader, epoch)
                
                # Log validation metrics
                for name, value in val_losses.items():
                    self.writer.add_scalar(f'Epoch/Val_{name}', value, epoch)
                
                # Print validation stats
                val_loss_str = ", ".join([f"{name}: {value:.4f}" for name, value in val_losses.items()])
                print(f'  Val: {val_loss_str}')
                
                # Save best model
                if save_path and val_losses.get('base_loss', float('inf')) < best_val_loss:
                    best_val_loss = val_losses.get('base_loss', float('inf'))
                    torch.save(self.model.state_dict(), save_path)
                    print(f'  Saved best model to {save_path}')
            
            # If no validation, save at regular intervals
            elif save_path and (epoch + 1) % 10 == 0:
                torch.save(self.model.state_dict(), f"{save_path}_epoch{epoch+1}")
                print(f'  Saved checkpoint to {save_path}_epoch{epoch+1}')
        
        print("Training complete!")
        return self.model

class AugmentedListDataset(Dataset):
    def __init__(self, Xs, cs, Ys, rotations=4):
        self.data = list(zip(Xs, cs, Ys))
        self.num_original = len(Xs)
        self.rotations = rotations  # Number of rotations (1=original, 2=+90°, 3=+180°, 4=+270°)
    
    def __len__(self):
        return self.num_original * self.rotations
    
    def __getitem__(self, idx):
        if idx >= self.num_original * self.rotations:
            raise IndexError()
        
        original_idx = idx % self.num_original
        rotation_idx = idx // self.num_original
        
        # Get original data
        assert(original_idx < self.num_original)
        X, c, Y = self.data[original_idx]
        
        # Apply rotation if needed
        if rotation_idx > 0:
            X = np.rot90(X, k=rotation_idx, axes=(0, 1)).copy()
            Y = np.rot90(Y, k=rotation_idx, axes=(0, 1)).copy()
        return X, Y, c


def prepare_dataset(dims=(20,20), repr_version=4,
                    center=True):
    F = load_dataset('av-redscience')
    errors = 0

    Xs = []
    Ys = []
    cs = []
    i = 0
    for f in F.factories.values():
        i += 1
        puncher = EntityPuncher(f)
        xs, ys, c = puncher.generate_state_action_pairs()
        for x, y, c_ in zip(xs, ys, c):
            try:
                x_m = x.get_matrix(dims=dims, repr_version=repr_version, center=center)
                y_m = y.get_matrix(dims=dims, repr_version=repr_version, center=center)
            except KeyError:
                logger.warning("Couldn't convert a matrix due to a NameError. Recording and continuing.")
                errors += 1
                continue
            Xs.append(x_m)
            Ys.append(y_m)
            cs.append(c_)
    return (Xs, cs, Ys)


def split_dataloader(dataloader, val_split=0.2, random_seed=42):
    """
    Split an existing DataLoader into training and validation DataLoaders.
    
    Args:
        dataloader: Existing DataLoader
        val_split: Fraction of data to use for validation (0.0 to 1.0)
        random_seed: Random seed for reproducibility
        
    Returns:
        train_loader, val_loader: DataLoader objects for training and validation
    """
    # Set random seed for reproducibility
    torch.manual_seed(random_seed)
    
    # Get the dataset from the dataloader
    dataset = dataloader.dataset
    
    # Calculate split sizes
    dataset_size = len(dataset)
    val_size = int(val_split * dataset_size)
    train_size = dataset_size - val_size
    
    # Split the dataset
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Create new DataLoaders with the same configuration as the original
    train_loader = DataLoader(
        train_dataset,
        batch_size=dataloader.batch_size,
        shuffle=True,
        collate_fn=collate_numpy_matrices,
        num_workers=dataloader.num_workers,
        pin_memory=dataloader.pin_memory,
        drop_last=dataloader.drop_last
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=dataloader.batch_size,
        shuffle=False,  # No need to shuffle validation data
        collate_fn=collate_numpy_matrices,
        num_workers=dataloader.num_workers,
        pin_memory=dataloader.pin_memory,
        drop_last=dataloader.drop_last
    )
    
    return train_loader, val_loader


if __name__ == "__main__":
    datalist = prepare_dataset()
    rotational_datalist = AugmentedListDataset(datalist)
    dataloader = DataLoader(
        rotational_datalist,
        batch_size=32, 
        collate_fn=collate_numpy_matrices
    )
    train_dataloader, val_dataloader = split_dataloader(dataloader)
    model = DeepQCNN(input_channels=8, output_channels=8)

    trainer = QCNNTrainer(
            model=model,
            device='mps',
            learning_rate=0.001,
            log_dir='runs/qcnn',
        )
    
    # Train model
    trainer.train(
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        num_epochs=100,
        save_path='models/qcnn_best.pth'
    )