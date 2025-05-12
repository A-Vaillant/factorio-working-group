import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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
from src.visualization import visualize_changes_for_tensorboard


logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)




# AV: This ended up being a huge boondoggle and I'm not using it.
def assembler_shape_loss(pred, asm_mask, weight):
    loss = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)

    # Create kernels that detect different parts of a 3x3 pattern
    kernel_3x3 = torch.ones((1, 1, 3, 3), device=pred.device, dtype=pred.dtype)
    
    # 1. Check every 3x3 patch
    # This operation returns a tensor where each value represents
    # the sum of a 3x3 region centered at that position
    conv_result = F.conv2d(
        F.pad(asm_mask, (1, 1, 1, 1), mode='constant', value=0),
        kernel_3x3,
        stride=1
    )
    
    # 2. Identify valid assembler regions
    # A valid region is where the 3x3 sum equals 9 (complete block)
    is_valid_center = (conv_result == 9).float()
    
    # 3. Any assembler pixel should be part of at least one valid 3x3 region
    # Dilate the valid centers to find all pixels that are part of valid blocks
    valid_region = F.conv2d(
        F.pad(is_valid_center, (1, 1, 1, 1), mode='constant', value=0),
        kernel_3x3,
        stride=1
    )
    
    # 4. Calculate the loss for assembler pixels not in any valid region
    # Only assembler pixels should be considered
    invalid_asm = asm_mask * F.relu(1.0 - (valid_region > 0).float())
    
    # Sum per batch then average
    shape_loss = invalid_asm.sum(dim=(1, 2, 3)).mean()
    
    loss += weight * shape_loss
    
    return loss


def matrix_integrity_loss(pred, weights=None):
    """
    Custom loss function to enforce structural integrity constraints on the 
    predicted matrix representation of Factorio blueprints.
    
    Args:
        pred: Tensor with shape [batch, channels, height, width]
        weights: dict to specify how much to punish certain things
        
    Returns:
        Loss tensor enforcing matrix integrity constraints
    """
    # Define channel indices based on the matrix structure
    channel_slices = {
        'assembler': slice(0, 1),
        'belt': slice(1, 2),
        'inserter': slice(2, 3),
        'pole': slice(3, 4),
        'direction': slice(4, 8),
        'recipe': slice(8, 13),
        'item': slice(13, 16),
        'kind': slice(16, 19),
        'sourcesink': slice(19, 21),
    }
    
    if weights is None:
        weights = {
            'onehot': 5.0,
            'coactivation': 1.0,
            'completeness': 3.0,
            'mutual_exclusion': 20.0,
            'shape': 4.0
        }
    weights = {k: torch.tensor(v, device=pred.device, dtype=pred.dtype) 
          for k, v in weights.items()}

    loss = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
    B, C, H, W = pred.shape
    
    def sum_channels(name):
        selected = pred[:, channel_slices[name], :, :]  # Should be (B, C', H, W)
        summed = selected.sum(1)  # Sum over channels, should give (B, H, W)
        return summed

    for field in ['direction', 'recipe', 'item', 'kind', 'sourcesink']:
        if field in channel_slices:
            ent_mask = sum_channels(field)
            onehot_violation = F.relu(ent_mask - 1)
            spatial_mean = onehot_violation.pow(2).mean(dim=[1, 2])  # [B]
            batch_mean = spatial_mean.mean()  # scalar
            loss = loss + (weights['onehot'] * batch_mean)

    # --- 2. Completeness constraints (entities must have certain fields) ---
    # Every entity must have an item ID (assembler, belt, inserter, pole)
    entity_fields = ['assembler', 'belt', 'inserter', 'pole']
    item_sum = sum_channels('item')
    for ent in entity_fields:
        ent_mask = sum_channels(ent)
        missing_item = F.relu(ent_mask - item_sum)
        loss = loss + (weights['completeness'] * missing_item.pow(2).mean())

    # Belts and inserters must have direction
    for ent in ['belt', 'inserter']:
        ent_mask = sum_channels(ent)
        dir_sum = sum_channels('direction')
        missing_dir = F.relu(ent_mask - dir_sum)
        loss += weights['completeness'] * missing_dir.pow(2).mean()

    # Belts must have kind
    if 'kind' in channel_slices:
        belt_mask = sum_channels('belt')
        kind_sum = sum_channels('kind')
        missing_kind = F.relu(belt_mask - kind_sum)
        loss += weights['completeness'] * missing_kind.pow(2).mean()

    # --- 3. Coactivation constraint: assembler's recipe ≤ assembler presence
    if 'recipe' in channel_slices and 'assembler' in channel_slices:
        asm_mask = sum_channels('assembler')
        recipe_sum = sum_channels('recipe')
        excess_recipe = F.relu(recipe_sum - asm_mask)
        loss += weights['coactivation'] * excess_recipe.pow(2).mean()

    # --- 4. Mutual exclusion: at most one entity per position
    ent_sum = sum(sum_channels(ent) for ent in entity_fields)
    overlap = F.relu(ent_sum - 1)
    loss += weights['mutual_exclusion'] * overlap.pow(2).mean()
    
    # --- Assembler shape enforcement (3x3 blocks) ---
    asm_mask = pred[:, channel_slices['assembler'], :, :]  # [B, 1, H, W]
    asm_counts = asm_mask.sum(dim=(1, 2, 3))  # [B]
    
    # Calculate the remainder when divided by 9
    # We use modulo in a differentiable way: asm_count - 9 * floor(asm_count / 9)
    # Penalize any non-zero remainder (not divisible by 9)
    # If remainder is 0, no penalty. If >0, penalty increases with size
    # Also penalize if remainder is very close to 9 (almost another complete assembler)
    # remainder = asm_counts - 9 * torch.floor(asm_counts / 9)
    # remainder_loss = torch.min(remainder, 9 - remainder).pow(2).mean()
    # loss += weights['shape'] * remainder_loss
    
    # asm_mask = pred[:, channel_slices['assembler'], :, :]
    # loss += assembler_shape_loss(pred, asm_mask, weights['shape'])
        
    return loss


def train_model(model, train_loader, val_loader, num_epochs=100, device='cuda',
                integrity_weight=0.1, log_dir=None, viz_interval=5):
    if log_dir is None:
        log_dir = f'runs/{model.filename}_{datetime.now().strftime("%Y%m%d-%H%M%S")}'
    # Make sure the models folder exists.
    os.makedirs('models', exist_ok=True)
    writer = SummaryWriter(log_dir)
    
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([15], device=device))
    # criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    if (device == 'cpu'):
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
    else:
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
            integrity_loss = matrix_integrity_loss(output)
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
        
        tp, tn, fp, fn = 0, 0, 0, 0
        with torch.no_grad():
            for i, (data, target, _) in enumerate(val_loader):
                data, target = data.to(device), target.to(device)
                output = model(data)
                
                # Store first batch for visualization
                if i == 0 and epoch % viz_interval == 0:
                    viz_data = (data[0], target[0], output[0])

                bce_loss = criterion(output, target.float())
                integrity_loss = matrix_integrity_loss(output)
                loss = bce_loss + integrity_weight * integrity_loss
                
                val_loss += loss.item()
                val_bce_loss += bce_loss.item()
                val_integrity_loss += integrity_loss.item()
                
                pred = (torch.sigmoid(output) > 0.5).float()
                # Calculate TP, TN, FP, FN
                tp += ((pred == 1) & (target == 1)).sum().item()
                tn += ((pred == 0) & (target == 0)).sum().item()
                fp += ((pred == 1) & (target == 0)).sum().item()
                fn += ((pred == 0) & (target == 1)).sum().item()
                correct += (pred == target).sum().item()
                total += target.numel()
        
        # Calculate average losses
        train_loss /= len(train_loader)
        train_bce_loss /= len(train_loader)
        train_integrity_loss /= len(train_loader)
        val_loss /= len(val_loader)
        val_bce_loss /= len(val_loader)
        val_integrity_loss /= len(val_loader)

        # Calculate metrics
        accuracy = 100. * correct / total
        tp_rate = 100. * tp / (tp + fn) if (tp + fn) > 0 else 0  # Sensitivity/Recall
        tn_rate = 100. * tn / (tn + fp) if (tn + fp) > 0 else 0  # Specificity
        precision = 100. * tp / (tp + fp) if (tp + fp) > 0 else 0
        f1_score = 2 * precision * tp_rate / (precision + tp_rate) if (precision + tp_rate) > 0 else 0

        # Log to tensorboard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/train_bce', train_bce_loss, epoch)
        writer.add_scalar('Loss/train_integrity', train_integrity_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Loss/val_bce', val_bce_loss, epoch)
        writer.add_scalar('Loss/val_integrity', val_integrity_loss, epoch)
        writer.add_scalar('Accuracy/val', accuracy, epoch)
        writer.add_scalar('TP_Rate/val', tp_rate, epoch)
        writer.add_scalar('TN_Rate/val', tn_rate, epoch)
        writer.add_scalar('Precision/val', precision, epoch)
        writer.add_scalar('F1_Score/val', f1_score, epoch)
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
        print(f'True Positive Rate: {tp_rate:.2f}% | True Negative Rate: {tn_rate:.2f}%')
        print(f'Precision: {precision:.2f}% | F1 Score: {f1_score:.2f}%')

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f'models/{model.filename}.pt')

        writer.close()


def test_model(model, test_loader, device='cpu',
               integrity_weight=0.1):
    model.eval()
    test_loss = 0
    test_bce_loss = 0
    test_integrity_loss = 0
    correct = 0
    total = 0
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([15], device=device))

    with torch.no_grad():
        for i, (data, target, *_) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)

            bce_loss = criterion(output, target.float())
            integrity_loss = matrix_integrity_loss(output)
            loss = bce_loss + integrity_weight * integrity_loss
            
            test_loss += loss.item()
            test_bce_loss += bce_loss.item()
            test_integrity_loss += integrity_loss.item()
            
            pred = (torch.sigmoid(output) > 0.5).float()
            correct += (pred == target).sum().item()
            total += target.numel()

    # Calculate average losses
    test_loss /= len(test_loader)
    test_bce_loss /= len(test_loader)
    test_integrity_loss /= len(test_loader)
    accuracy = 100. * correct / total

    # Removed tensorboard logging
    print(f'Test Loss: {test_loss:.4f} (BCE: {test_bce_loss:.4f}, Integrity: {test_integrity_loss:.4f})')
    print(f'Test Accuracy: {accuracy:.2f}%')
    
    
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
    return (Ys, cs, Xs)  # Switch before and after factories HERE.


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