import matplotlib.pyplot as plt
import numpy as np
import torch

channel_list = ['assembler', 'inserter', 'belt', 'pole', 'direction', 'power']
channel_names = [
    'assemblers', 'belts', 'inserters', 'poles'
]

dirs = ['left', 'top', 'right', 'bottom']
channel_names += [f'dir: {d}' for d in dirs]
channel_names += [f'recipe #{i+1}' for i in range(5)]
channel_names += [f'item tier #{i+1}' for i in range(3)]
channel_names += ['standard belts', 'underground belts', 'splitters']
channel_names += ['sources', 'sinks']


def visualize_6D_matrix(matrix, hwc=None):
    """
    Visualize each channel of a multichannel 2D matrix in separate subplots.
    
    Parameters:
    -----------
    matrix : numpy.ndarray
        A matrix with shape (channels, height, width) or (height, width, channels)
    """
    # Determine the shape and reorder if necessary
    if matrix.ndim != 3:
        raise ValueError("Input must be a 3D array")
    
    # Check if channels are first or last dimension

    if hwc == False or matrix.shape[0] == 6:
        # (channels, height, width) format
        channels, height, width = matrix.shape
    elif hwc or matrix.shape[2] == 6:
        # (height, width, channels) format
        height, width, channels = matrix.shape
        # Transpose to (channels, height, width)
        matrix = np.transpose(matrix, (2, 0, 1))
    else:
        raise ValueError("Input must have exactly 6 channels")
    
    # Create figure and subplots with specific spacing
    fig, axes = plt.subplots(2, 3, figsize=(15, 10), 
                             gridspec_kw={'wspace': 0.3, 'hspace': 0.3})
    axes = axes.flatten()
    
    # Plot each channel with exact pixel representation
    for i in range(channels):
        # Turn off interpolation for exact pixel representation
        im = axes[i].imshow(matrix[i], cmap='viridis', 
                           interpolation='none', aspect='equal')
        
        # Add precise grid lines at each pixel boundary
        height, width = matrix[i].shape
        
        # Draw vertical grid lines
        for x in range(width + 1):
            axes[i].axvline(x - 0.5, color='black', linewidth=0.5, alpha=0.5)
            
        # Draw horizontal grid lines
        for y in range(height + 1):
            axes[i].axhline(y - 0.5, color='black', linewidth=0.5, alpha=0.5)
            
        # Set ticks at each integer position
        axes[i].set_xticks(np.arange(width))
        axes[i].set_yticks(np.arange(height))
        
        # Hide tick labels for cleaner appearance
        axes[i].set_xticklabels([])
        axes[i].set_yticklabels([])
        
        # Make sure the limits are exactly at pixel boundaries
        axes[i].set_xlim(-0.5, width - 0.5)
        axes[i].set_ylim(height - 0.5, -0.5)  # Note: y-axis is inverted in imshow
        
        # Add border
        for spine in axes[i].spines.values():
            spine.set_visible(True)
            spine.set_color('black')
            spine.set_linewidth(1)
            
        axes[i].set_title(f'{channel_list[i]}', fontsize=12, pad=10)
    
    plt.tight_layout()
    plt.suptitle('6-Channel Matrix Visualization', fontsize=16)
    plt.subplots_adjust(top=0.9)  # Make room for the title
    
    return fig


def visualize_7D_matrix(matrix, channel_list=None, hwc=None):
    """
    Visualize each channel of a multichannel 2D matrix in separate subplots.
   
    Parameters:
    -----------
    matrix : numpy.ndarray
        A matrix with shape (channels, height, width) or (height, width, channels)
    channel_list : list of str, optional
        Names for each channel to display as titles
    hwc : bool, optional
        If True, assumes input is (height, width, channels)
        If False, assumes input is (channels, height, width)
        If None, tries to determine automatically
    """
    # Default channel names if not provided
    if channel_list is None:
        channel_list = ["Assembler", "Inserter", "Belt", "Pole", 
                        "Direction", "Recipe", "Item"]
    
    # Determine the shape and reorder if necessary
    if matrix.ndim != 3:
        raise ValueError("Input must be a 3D array")
   
    # Check if channels are first or last dimension
    if hwc == False or matrix.shape[0] == 7:
        # (channels, height, width) format
        channels, height, width = matrix.shape
    elif hwc or matrix.shape[2] == 7:
        # (height, width, channels) format
        height, width, channels = matrix.shape
        # Transpose to (channels, height, width)
        matrix = np.transpose(matrix, (2, 0, 1))
    else:
        raise ValueError("Input must have exactly 7 channels")
   
    # Create figure and subplots with specific spacing
    fig, axes = plt.subplots(3, 3, figsize=(15, 15),
                             gridspec_kw={'wspace': 0.3, 'hspace': 0.3})
    axes = axes.flatten()
   
    # Plot each channel with exact pixel representation
    for i in range(channels):
        # Turn off interpolation for exact pixel representation
        im = axes[i].imshow(matrix[i], cmap='viridis',
                           interpolation='none', aspect='equal')
       
        # Add colorbar for reference
        plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
        
        # Add precise grid lines at each pixel boundary
        height, width = matrix[i].shape
       
        # Draw vertical grid lines
        for x in range(width + 1):
            axes[i].axvline(x - 0.5, color='black', linewidth=0.5, alpha=0.5)
           
        # Draw horizontal grid lines
        for y in range(height + 1):
            axes[i].axhline(y - 0.5, color='black', linewidth=0.5, alpha=0.5)
           
        # Set ticks at each integer position
        axes[i].set_xticks(np.arange(width))
        axes[i].set_yticks(np.arange(height))
       
        # Hide tick labels for cleaner appearance
        axes[i].set_xticklabels([])
        axes[i].set_yticklabels([])
       
        # Make sure the limits are exactly at pixel boundaries
        axes[i].set_xlim(-0.5, width - 0.5)
        axes[i].set_ylim(height - 0.5, -0.5)  # Note: y-axis is inverted in imshow
       
        # Add border
        for spine in axes[i].spines.values():
            spine.set_visible(True)
            spine.set_color('black')
            spine.set_linewidth(1)
           
        axes[i].set_title(f'{channel_list[i]}', fontsize=12, pad=10)
   
    # Hide the unused subplot (9 plots for 7 channels)
    for i in range(channels, 9):
        axes[i].axis('off')
   
    plt.tight_layout()
    plt.suptitle('7-Channel Matrix Visualization', fontsize=16)
    plt.subplots_adjust(top=0.9)  # Make room for the title
   
    return fig


def print_matrix_channels(matrix, channel_names=None):
    """
    Print each channel of a multi-channel matrix in a readable format.
    
    Args:
        matrix: Numpy array with shape (width, height, channels)
        channel_names: Optional list of names for each channel
    """
    if channel_names is None:
        channel_names = [f"Channel {i}" for i in range(matrix.shape[2])]
    
    width, height, channels = matrix.shape
    
    for c in range(channels):
        print(f"\n=== {channel_names[c]} ===")
        
        # Print column headers
        print("    ", end="")
        for x in range(width):
            print(f"{x:2d} ", end="")
        print()
        
        # Print matrix with row headers
        for y in range(height):
            print(f"{y:2d}  ", end="")
            for x in range(width):
                val = matrix[x, y, c]
                if val == 0:
                    print(" . ", end="")
                else:
                    print(f"{val:2d} ", end="")
            print()


def visualize_8channel_comparison(inputs, targets, outputs, num_samples=1, cmap='viridis'):
    """
    Visualize a comparison of 8-channel data for inputs, targets, and outputs.
    
    Args:
        inputs: Input tensor of shape [batch_size, 8, height, width]
        targets: Target tensor of shape [batch_size, 8, height, width]
        outputs: Output tensor of shape [batch_size, 8, height, width]
        num_samples: Number of samples to visualize
        cmap: Colormap to use for visualization
    """
    # Ensure all tensors are on CPU and converted to numpy
    if torch.is_tensor(inputs):
        inputs = inputs.detach().cpu().numpy()
    if torch.is_tensor(targets):
        targets = targets.detach().cpu().numpy()
    if torch.is_tensor(outputs):
        outputs = outputs.detach().cpu().numpy()
    
    for sample_idx in range(min(num_samples, inputs.shape[0])):
        # Create a 3x8 grid plot
        fig, axes = plt.subplots(3, 8, figsize=(20, 7))
        plt.subplots_adjust(wspace=0.1, hspace=0.2)
        
        # Set title for the figure
        fig.suptitle(f'Sample {sample_idx+1}: Channel Comparison', fontsize=16)
        
        # Row labels
        row_labels = ['Input', 'Ground Truth', 'Output']
        
        # For each channel
        for channel in range(8):
            # Input channel
            im1 = axes[0, channel].imshow(inputs[sample_idx, channel], cmap=cmap)
            axes[0, channel].set_title(f'Ch {channel+1}')
            axes[0, channel].axis('off')
            
            # Ground truth channel
            im2 = axes[1, channel].imshow(targets[sample_idx, channel], cmap=cmap)
            axes[1, channel].axis('off')
            
            # Output channel
            im3 = axes[2, channel].imshow(outputs[sample_idx, channel], cmap=cmap)
            axes[2, channel].axis('off')
            
            # Add row labels on the left side
            if channel == 0:
                for row_idx, label in enumerate(row_labels):
                    axes[row_idx, 0].set_ylabel(label, fontsize=12, rotation=90)
        
        # Add colorbar
        cbar_ax = fig.add_axes([0.92, 0.15, 0.01, 0.7])
        fig.colorbar(im1, cax=cbar_ax)
        
        plt.tight_layout(rect=[0, 0, 0.9, 0.95])
        plt.show()


def visualize_manychannel_matrices(input_matrix, ground_truth, output_matrix, save_path=None):
    """
    Visualize 21-channel binary matrices side by side.
    
    Args:
        input_matrix: torch.Tensor of shape [21, 20, 20]
        ground_truth: torch.Tensor of shape [21, 20, 20]
        output_matrix: torch.Tensor of shape [21, 20, 20]
        save_path: Optional path to save the figure
    """
    # Convert to numpy and ensure binary values
    input_np = input_matrix.permute(1, 2, 0).cpu().detach().numpy()
    ground_truth_np = ground_truth.permute(1, 2, 0).cpu().detach().numpy()
    output_np = (torch.sigmoid(output_matrix) > 0.5).permute(1, 2, 0).cpu().detach().numpy()

    # Calculate max number of non-empty channels
    n_channels = 0
    for row in range(21):
        if (input_np[:,:,row].any() or 
            ground_truth_np[:,:,row].any() or 
            output_np[:,:,row].any()):
            n_channels += 1
    
    if n_channels == 0:
        print("All channels are empty!")
        return
    
    fig, axes = plt.subplots(21, 3, figsize=(10, n_channels * 7))
    plt.subplots_adjust(hspace=0.1, wspace=0.1)

    titles = ['Input', 'Ground Truth', 'Prediction']
    
    # Set column titles
    for col, title in enumerate(titles):
        axes[0, col].set_title(title)
    
    forest_green = (0.133, 0.545, 0.133)
    bright_red = (1, 0, 0)         # Pure bright red for false positives
    dark_red = (0.5, 0, 0.5)       # Purple for false negatives

    current_row = 0
    for row in range(21):
        # Skip if channel is empty
        if not (input_np[:,:,row].any() or 
                ground_truth_np[:,:,row].any() or 
                output_np[:,:,row].any()):
            continue
            
        # Show input and ground truth
        axes[current_row, 0].imshow(input_np[:,:,row], cmap='binary', interpolation='nearest')
        axes[current_row, 1].imshow(ground_truth_np[:,:,row], cmap='binary', interpolation='nearest')
        
        # Create white background
        change_viz = np.ones((*input_np.shape[:2], 3))
        
        # Set unchanged 1s to black
        unchanged = (input_np[:,:,row] == output_np[:,:,row])
        unchanged_ones = unchanged & (input_np[:,:,row] == 1)
        change_viz[unchanged_ones] = [0, 0, 0]
        
        # Separate false positives and false negatives
        false_positives = (output_np[:,:,row] == 1) & (ground_truth_np[:,:,row] == 0)
        false_negatives = (output_np[:,:,row] == 0) & (ground_truth_np[:,:,row] == 1)
        
        # Color the changes
        correct_changes = (input_np[:,:,row] != output_np[:,:,row]) & (output_np[:,:,row] == ground_truth_np[:,:,row])
        
        # Green = correct changes
        change_viz[correct_changes] = forest_green
        # Bright red = false positives (0->1 wrongly)
        change_viz[false_positives] = bright_red
        # Dark red = false negatives (1->0 wrongly)
        change_viz[false_negatives] = dark_red
        
        axes[current_row, 2].imshow(change_viz, interpolation='nearest')
        
        # Clean up axes
        for col in range(3):
            axes[current_row, col].set_xticks([])
            axes[current_row, col].set_yticks([])
            if col == 0:
                axes[current_row, col].set_ylabel(f'{channel_names[row]}')
        
        current_row += 1
    
    # Remove empty subplots
    for row in range(current_row, 21):
        for col in range(3):
            fig.delaxes(axes[row, col])
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.show()

def visualize_changes_for_tensorboard(writer, epoch, input_matrix, ground_truth, output_matrix):
    """
    Log matrix visualizations to tensorboard showing:
    - Original input
    - Ground truth
    - Changes (green = correct predictions, red = incorrect predictions)
    """
    # Convert to numpy and ensure binary values
    input_np = input_matrix.permute(1, 2, 0).cpu().detach().numpy()
    ground_truth_np = ground_truth.permute(1, 2, 0).cpu().detach().numpy()
    output_np = (torch.sigmoid(output_matrix) > 0.5).permute(1, 2, 0).cpu().detach().numpy()
    
    # Create figure
    fig, axes = plt.subplots(21, 3, figsize=(15, 80))
    titles = ['Input', 'Ground Truth', 'Changes (Green=Correct, Red=Wrong)']
    
    # Set column titles
    for col, title in enumerate(titles):
        axes[0, col].set_title(title, pad=20)
    
    forest_green = (0.133, 0.545, 0.133)

    for row in range(21):
        # Show input and ground truth
        axes[row, 0].imshow(input_np[:,:,row], cmap='binary', interpolation='nearest')
        axes[row, 1].imshow(ground_truth_np[:,:,row], cmap='binary', interpolation='nearest')
        
        # Create white background
        change_viz = np.ones((*input_np.shape[:2], 3))
        
        # Set unchanged 1s to black
        unchanged = (input_np[:,:,row] == output_np[:,:,row])
        unchanged_ones = unchanged & (input_np[:,:,row] == 1)
        change_viz[unchanged_ones] = [0, 1, 1]
        
        # Color the changes
        changes = (input_np[:,:,row] != output_np[:,:,row])
        correct_changes = changes & (output_np[:,:,row] == ground_truth_np[:,:,row])
        incorrect_changes = changes & (output_np[:,:,row] != ground_truth_np[:,:,row])
        
        # Green = correct changes
        change_viz[correct_changes] = forest_green
        # Red = incorrect changes
        change_viz[incorrect_changes] = [1, 0, 0]
        
        axes[row, 2].imshow(change_viz, interpolation='nearest')
        
        # Clean up axes
        for col in range(3):
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
            if col == 0:
                axes[row, col].set_ylabel(channel_names[row])
    
    plt.tight_layout()
    writer.add_figure('Matrix_Visualization', fig, epoch)
    plt.close()
    
    
def visualize_factory_matrix(entity_matrices):
    """Convert entity matrices into a visual representation using Unicode characters.
    
    Symbol mapping:
    Assemblers (recipes 1-5): ⬣ ⬢ ⬡ ⬥ ⬦
    Belts:
        Normal:    → ↓ ← ↑
        Underground: ⇒ ⇓ ⇐ ⇑
        Splitter:   ⇢ ⇣ ⇠ ⇡
    Inserters:     ▻ ▾ ◅ ▴
    Poles:         ⚡
    Empty:         ·
    """
    h, w = next(iter(entity_matrices.values())).shape
    visual_matrix = np.full((h, w), '·', dtype=str)
    
    # Character mappings
    assembler_chars = ['·', '⬣', '⬢', '⬡', '⬥', '⬦']
    belt_normal = ['·', '→', '↓', '←', '↑']
    belt_underground = ['·', '⇒', '⇓', '⇐', '⇑']
    belt_splitter = ['·', '⇢', '⇣', '⇠', '⇡']
    inserter_chars = ['·', '▻', '▾', '◅', '▴']
    pole_chars = ['·', '⚡']
    
    # Layer the entities from back to front
    # First poles (background)
    pole_mask = entity_matrices['pole'] > 0
    visual_matrix[pole_mask] = pole_chars[1]
    
    # Then belts
    belt_matrix = entity_matrices['belt']
    belt_mask = belt_matrix > 0
    for i in range(belt_mask.shape[0]):
        for j in range(belt_mask.shape[1]):
            if belt_matrix[i,j] > 0:
                direction = ((belt_matrix[i,j] - 1) % 4) + 1
                kind = (belt_matrix[i,j] - 1) // 4
                if kind == 0:
                    visual_matrix[i,j] = belt_normal[direction]
                elif kind == 1:
                    visual_matrix[i,j] = belt_underground[direction]
                else:
                    visual_matrix[i,j] = belt_splitter[direction]
    
    # Then inserters
    inserter_matrix = entity_matrices['inserter']
    inserter_mask = inserter_matrix > 0
    for i in range(inserter_mask.shape[0]):
        for j in range(inserter_mask.shape[1]):
            if inserter_matrix[i,j] > 0:
                visual_matrix[i,j] = inserter_chars[inserter_matrix[i,j]]
    
    # Finally assemblers (foreground)
    assembler_matrix = entity_matrices['assembler']
    assembler_mask = assembler_matrix > 0
    for i in range(assembler_mask.shape[0]):
        for j in range(assembler_mask.shape[1]):
            if assembler_matrix[i,j] > 0:
                visual_matrix[i,j] = assembler_chars[assembler_matrix[i,j]]
    
    return visual_matrix

def print_factory(visual_matrix):
    """Print the visual matrix with proper spacing."""
    for row in visual_matrix.T:
        print(' '.join(row))