import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

from src.visualization import visualize_changes_for_tensorboard


class BinaryMatrixTransformCNN(nn.Module):
    def __init__(self, channels=21, matrix_size=20):
        super(BinaryMatrixTransformCNN, self).__init__()
        
        # Multiple conv layers with padding to maintain spatial dimensions
        self.conv1 = nn.Conv2d(channels, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        # Project back to original dimensions
        self.conv_out = nn.Conv2d(128, channels, kernel_size=1)
        
    def forward(self, x):
        # Input shape: [batch, 21, 20, 20]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.conv_out(x)
        # Output shape: [batch, 21, 20, 20]
        return x

    def predict(self, X):
        """Make a prediction with thresholding"""
        self.eval()
        with torch.no_grad():
            # Get raw outputs
            raw_outputs = self(X)
            
            # Apply thresholding
            thresholded_outputs = (raw_outputs > 0.5).float()
            
            return thresholded_outputs
        

def matrix_integrity_loss(output, target):
    # Placeholder for your technical matrix integrity calculation
    # Return a scalar tensor
    return torch.tensor(0.0, device=output.device, requires_grad=True)

def train_model(model, train_loader, val_loader, num_epochs=100, device='cuda',
                integrity_weight=0.1, log_dir=None, viz_interval=5):
    if log_dir is None:
        log_dir = f'runs/binary_matrix_transform_{datetime.now().strftime("%Y%m%d-%H%M%S")}'
    writer = SummaryWriter(log_dir)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    model = model.to(device)
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        train_bce_loss = 0
        train_integrity_loss = 0
        
        for batch_idx, (data, target, _) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            
            bce_loss = criterion(output, target.float())
            integrity_loss = matrix_integrity_loss(output, target)
            loss = bce_loss + integrity_weight * integrity_loss
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_bce_loss += bce_loss.item()
            train_integrity_loss += integrity_loss.item()
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_bce_loss = 0
        val_integrity_loss = 0
        correct = 0
        total = 0

        viz_data = None
        
        with torch.no_grad():
            for i, (data, target, _) in enumerate(val_loader):
                data, target = data.to(device), target.to(device)
                output = model(data)
                
                # Store first batch for visualization
                if i == 0 and epoch % viz_interval == 0:
                    viz_data = (data[0], target[0], output[0])

                bce_loss = criterion(output, target.float())
                integrity_loss = matrix_integrity_loss(output, target)
                loss = bce_loss + integrity_weight * integrity_loss
                
                val_loss += loss.item()
                val_bce_loss += bce_loss.item()
                val_integrity_loss += integrity_loss.item()
                
                pred = (torch.sigmoid(output) > 0.5).float()
                correct += (pred == target).sum().item()
                total += target.numel()
        
        # Calculate average losses
        train_loss /= len(train_loader)
        train_bce_loss /= len(train_loader)
        train_integrity_loss /= len(train_loader)
        val_loss /= len(val_loader)
        val_bce_loss /= len(val_loader)
        val_integrity_loss /= len(val_loader)
        accuracy = 100. * correct / total
        
        # Log to tensorboard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/train_bce', train_bce_loss, epoch)
        writer.add_scalar('Loss/train_integrity', train_integrity_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Loss/val_bce', val_bce_loss, epoch)
        writer.add_scalar('Loss/val_integrity', val_integrity_loss, epoch)
        writer.add_scalar('Accuracy/val', accuracy, epoch)
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)

        # Add matrix visualization every viz_interval epochs
        if epoch % viz_interval == 0 and viz_data is not None:
            input_matrix, target_matrix, output_matrix = viz_data
            visualize_changes_for_tensorboard(writer, epoch, 
                                           input_matrix, target_matrix, output_matrix)
        
        print(f'Epoch: {epoch+1}')
        print(f'Training Loss: {train_loss:.4f} (BCE: {train_bce_loss:.4f}, Integrity: {train_integrity_loss:.4f})')
        print(f'Validation Loss: {val_loss:.4f} (BCE: {val_bce_loss:.4f}, Integrity: {val_integrity_loss:.4f})')
        print(f'Validation Accuracy: {accuracy:.2f}%')
        
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pt')
    
    writer.close()