from torch.utils.data import DataLoader

from src.pipeline.loaders import FactoryLoader, MatrixDataset, collate_numpy_matrices
from src.model import DeepQCNN
from src.pipeline.filters import (Required,
                                  RecipeWhitelist,
                                  SizeRestrictor,
                                  Blacklist,
                                  Whitelist)


# We probably have the PoD stuff around here.
# Determine what factories we're going to use.
F = FactoryLoader
M = MatrixDataset


dataloader = DataLoader(
        M, 
        batch_size=32, 
        collate_fn=collate_numpy_matrices
)
