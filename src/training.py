import numpy as np
from torch.utils.data import Dataset, DataLoader

from src.pipeline.loaders import FactoryLoader, MatrixDataset, collate_numpy_matrices
from src.model import DeepQCNN
from src.pipeline import load_dataset
from src.pipeline.filters import (Required,
                                  RecipeWhitelist,
                                  SizeRestrictor,
                                  Blacklist,
                                  Whitelist)


# We probably have the PoD stuff around here.
# Determine what factories we're going to use.
F = FactoryLoader
M = MatrixDataset


class AugmentedListDataset(Dataset):
    def __init__(self, data_list, rotations=4):
        self.data = data_list
        self.num_original = len(data_list)
        self.rotations = rotations  # Number of rotations (1=original, 2=+90°, 3=+180°, 4=+270°)
    
    def __len__(self):
        return self.num_original * self.rotations
    
    def __getitem__(self, idx):
        # Determine which original sample and which rotation
        original_idx = idx % self.num_original
        rotation_idx = idx // self.num_original
        
        # Get original data
        X, c, Y = self.data[original_idx]
        
        # Apply rotation if needed
        if rotation_idx > 0:
            # Rotate X and Y by k*90 degrees
            X = np.rot90(X, k=rotation_idx, axes=(0, 1))
            Y = np.rot90(Y, k=rotation_idx, axes=(0, 1))
            
            # You might need to adjust 'c' based on the rotation
            # if 'c' represents something that changes with rotation
            # c = transform_label_with_rotation(c, rotation_idx)
        
        return X, c, Y
    
    
dataloader = DataLoader(
        M, 
        batch_size=32, 
        collate_fn=collate_numpy_matrices
)


def main():
    # Load blueprint dataset
    dataset = load_dataset("av-redscience")
    k, factory = next(iter(dataset.factories.items()))
    ep = EntityPuncher(factory)
    X_before, X_after, y = ep.generate_state_action_pairs(num_pairs=10000)

    # Set up model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = QNetwork().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    epochs = 5
    losses = []

    for epoch in range(epochs):
        total_loss = 0.0
        for i in range(len(X_before)):
            state_np = X_before[i].transpose(2, 0, 1)  # (4, 20, 20)
            next_state_np = X_after[i].transpose(2, 0, 1)
            ch = y[i]

            diff = next_state_np[ch] - state_np[ch]
            yx = np.argwhere(diff == 1)
            if len(yx) == 0:
                continue  # Skip if no visible change
            y_idx, x_idx = yx[0]
            action_index = (y_idx * 20 + x_idx) * 4 + ch

            state_tensor = torch.tensor(state_np, dtype=torch.float32).unsqueeze(0).to(device)
            q_values = model(state_tensor)  # (1, 1600)

            # Supervised Q-learning target
            target_q = q_values.clone().detach()
            target_q[0, action_index] = 1.0  # Reward for correct action

            loss = F.mse_loss(q_values, target_q)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(X_before)
        losses.append(avg_loss)
        print(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.6f}")


if __name__ == "__main__":
    main()
