import unittest
import numpy as np

import torch
from torch.utils.data import DataLoader
from src.training import AugmentedListDataset

import unittest
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

# Import these from your implementation file
# from your_module import AugmentedListDataset, collate_numpy_matrices

class AugmentedListDataset(Dataset):
    def __init__(self, Xs, cs, Ys, rotations=4):
        self.data = list(zip(Xs, cs, Ys))
        self.num_original = len(Xs)
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
            X = np.rot90(X, k=rotation_idx)
            Y = np.rot90(Y, k=rotation_idx)
            
        return X, c, Y


class TestAugmentedListDataset(unittest.TestCase):
    def setUp(self):
        self.Xs = [
            np.array([
                [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                [[10, 20, 30], [40, 50, 60], [70, 80, 90]],
                [[12, 13, 14], [15, 16, 17], [18, 19, 20]]
            ]),
            np.array([
                [[11, 12, 13], [14, 15, 16], [17, 18, 19]],
                [[21, 22, 23], [24, 25, 26], [27, 28, 29]],
                [[12, 13, 14], [15, 16, 17], [18, 19, 20]]
            ]),
            np.array([
                [[31, 32, 33], [34, 35, 36], [37, 38, 39]],
                [[41, 42, 43], [44, 45, 46], [47, 48, 49]],
                [[12, 13, 14], [15, 16, 17], [18, 19, 20]]
            ])
        ]
        
        self.cs = [100, 200, 300]
        
        self.Ys = [
            np.array([
                [[101, 102, 103], [104, 105, 106], [107, 108, 109]],
                [[110, 120, 130], [140, 150, 160], [170, 180, 190]],
                [[101, 102, 103], [104, 105, 106], [107, 108, 109]]
            ]),
            np.array([
                [[111, 112, 113], [114, 115, 116], [117, 118, 119]],
                [[121, 122, 123], [124, 125, 126], [127, 128, 129]],
                [[101, 102, 103], [104, 105, 106], [107, 108, 109]]
            ]),
            np.array([
                [[131, 132, 133], [134, 135, 136], [137, 138, 139]],
                [[141, 142, 143], [144, 145, 146], [147, 148, 149]],
                [[101, 102, 103], [104, 105, 106], [107, 108, 109]]
            ])
        ]
    
    def test_dataset_length(self):
        # Test with default rotations (4)
        dataset = AugmentedListDataset(self.Xs, self.cs, self.Ys)
        self.assertEqual(len(dataset), len(self.Xs) * 4)
        
        # Test with custom rotations
        dataset = AugmentedListDataset(self.Xs, self.cs, self.Ys, rotations=2)
        self.assertEqual(len(dataset), len(self.Xs) * 2)
        
        # Test with no rotations
        dataset = AugmentedListDataset(self.Xs, self.cs, self.Ys, rotations=1)
        self.assertEqual(len(dataset), len(self.Xs))
    
    def test_accessibility(self):
        dataset = AugmentedListDataset(self.Xs, self.cs, self.Ys)
        for ix in range(len(dataset)):
            # print(f"Reached number {ix}.")
            self.assertIsNotNone(dataset[ix])
            self.assertLess(ix, len(dataset))

    def test_expected_orig(self):
        dataset = AugmentedListDataset(self.Xs, self.cs, self.Ys)
        self.assertEqual(dataset.num_original, len(self.Xs))

    def test_length_expectation(self):
        dataset = AugmentedListDataset(self.Xs, self.cs, self.Ys)
        L = len(dataset)
        with self.assertRaises(IndexError):
            dataset[15]
            dataset[L]
    

if __name__ == '__main__':
    unittest.main()