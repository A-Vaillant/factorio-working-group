import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from pathlib import Path
import sys

# Add src to path if needed
sys.path.append(str(Path.cwd() / "src"))

from syntheticdata.processor import EntityPuncher, load_dataset


class QNetwork(nn.Module):
    def __init__(self, in_channels=4, grid_size=20, num_channels=4):
        super().__init__()
        self.grid_size = grid_size
        self.num_actions = grid_size * grid_size * num_channels
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * grid_size * grid_size, 512),
            nn.ReLU(),
            nn.Linear(512, self.num_actions)
        )

    def forward(self, x):
        return self.net(x)


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
