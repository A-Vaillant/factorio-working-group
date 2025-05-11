import torch
import numpy as np
import unittest

from src.training import matrix_integrity_loss
from src.representation.factory import Factory


import torch
import numpy as np
import unittest

class TestMatrixIntegrityLoss(unittest.TestCase):
    def setUp(self):
        # Set random seed for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)
        self.batch_size = 2
        self.height = 10
        self.width = 10
        
        # Correct channel count to 21
        self.num_channels = 21
        
        # Define channel indices based on the code from the paste
        # Opacity channels (first 4 channels)
        self.assembler_ch = 0
        self.belt_ch = 1
        self.inserter_ch = 2
        self.pole_ch = 3
        
        # Other channels
        # Based on the order in json_to_manychannel_matrix
        self.direction_ch = 4  # 4 channels for direction
        self.recipe_ch = 8     # 5 channels for recipe
        self.item_ch = 13      # 3 channels for item types
        self.kind_ch = 16      # 3 channels for belt kinds
        self.sourcesink_ch = 19  # 2 channels for sourcesink
        
    def test_no_overlap_no_loss(self):
        """Test that non-overlapping entities produce no overlap loss."""
        pred_matrix = torch.zeros(self.batch_size, self.num_channels, self.height, self.width)
        
        # Create valid assemblers (3x3 blocks) with no overlap
        # First batch, first assembler - 3x3 block
        pred_matrix[0, self.assembler_ch, 1:4, 1:4] = 1.0  # Assembler opacity
        pred_matrix[0, self.recipe_ch, 1:4, 1:4] = 1.0     # Recipe channel
        pred_matrix[0, self.item_ch, 1:4, 1:4] = 1.0       # Item type (tier 1)
        
        # Second batch, second assembler at different position
        pred_matrix[1, self.assembler_ch, 5:8, 5:8] = 1.0  # Assembler opacity
        pred_matrix[1, self.recipe_ch+1, 5:8, 5:8] = 1.0   # Different recipe
        pred_matrix[1, self.item_ch+1, 5:8, 5:8] = 1.0     # Item type (tier 2)
        
        # Add some belts with no overlap
        pred_matrix[0, self.belt_ch, 5, 5] = 1.0           # Belt opacity
        pred_matrix[0, self.direction_ch, 5, 5] = 1.0      # Belt direction (first direction)
        pred_matrix[0, self.item_ch, 5, 5] = 1.0           # Belt item type
        pred_matrix[0, self.kind_ch, 5, 5] = 1.0           # Belt kind (normal belt)
        
        pred_matrix[1, self.belt_ch, 2, 8] = 1.0           # Belt opacity
        pred_matrix[1, self.direction_ch+1, 2, 8] = 1.0    # Belt direction (second direction)
        pred_matrix[1, self.item_ch, 2, 8] = 1.0           # Belt item type
        pred_matrix[1, self.kind_ch, 2, 8] = 1.0           # Belt kind (normal belt)
        
        loss = matrix_integrity_loss(pred_matrix)
        
        # Since there's no overlap and valid entities, loss should be very low
        self.assertLess(loss.item(), 0.01)
        
    def test_overlap_produces_loss(self):
        """Test that overlapping entities produce high loss."""
        pred_matrix = torch.zeros(self.batch_size, self.num_channels, self.height, self.width)
        
        # Create overlapping entities
        # Assembler and belt in same position
        pred_matrix[0, self.assembler_ch, 3:6, 3:6] = 1.0  # Assembler opacity
        pred_matrix[0, self.belt_ch, 4:5, 4:5] = 1.0       # Belt opacity overlapping with assembler
        
        loss = matrix_integrity_loss(pred_matrix)
        
        # Should have significant loss due to overlap
        self.assertGreater(loss.item(), 0.1)
        
    def test_incomplete_assembler_produces_loss(self):
        """Test that incomplete assemblers (not 3x3) produce high loss."""
        pred_matrix = torch.zeros(self.batch_size, self.num_channels, self.height, self.width)
        
        # Create incomplete assembler (2x3 instead of 3x3)
        pred_matrix[0, self.assembler_ch, 2:4, 2:5] = 1.0  # Assembler opacity
        
        loss = matrix_integrity_loss(pred_matrix)
        
        # Should have significant loss due to incomplete assembler
        self.assertGreater(loss.item(), 0.1)
        
    def test_belt_direction_consistency(self):
        """Test that belts with inconsistent directions produce high loss."""
        pred_matrix = torch.zeros(self.batch_size, self.num_channels, self.height, self.width)
        
        # Belt with no direction
        pred_matrix[0, self.belt_ch, 3, 3] = 1.0           # Belt opacity
        
        # Belt with multiple directions
        pred_matrix[0, self.belt_ch, 5, 5] = 1.0           # Belt opacity
        pred_matrix[0, self.direction_ch, 5, 5] = 1.0      # First direction
        pred_matrix[0, self.direction_ch+1, 5, 5] = 1.0    # Second direction - inconsistent!
        
        loss = matrix_integrity_loss(pred_matrix)
        
        # Should have significant loss due to direction inconsistency
        self.assertGreater(loss.item(), 0.1)
        
    def test_pole_size_constraints(self):
        """Test that incorrectly sized poles produce high loss."""
        pred_matrix = torch.zeros(self.batch_size, self.num_channels, self.height, self.width)
        
        # Large pole (indexes 2-3 in pole item channels) should be 2x2 but is 1x2
        # Using the third pole type (index 2)
        pred_matrix[0, self.pole_ch, 2:3, 2:4] = 1.0       # Pole opacity
        pred_matrix[0, self.item_ch+2, 2:3, 2:4] = 1.0     # Large pole type
        
        loss = matrix_integrity_loss(pred_matrix)
        
        # Should have loss due to incomplete large pole
        self.assertGreater(loss.item(), 0.05)
        
    def test_gradient_flow(self):
        """Test that gradients flow properly through the loss function."""
        pred_matrix = torch.zeros(self.batch_size, self.num_channels, self.height, self.width, 
                                 requires_grad=True)
        
        # Create some overlapping entities to generate loss
        pred_matrix.data[0, self.assembler_ch, 3:6, 3:6] = 0.6  # Assembler
        pred_matrix.data[0, self.belt_ch, 4:5, 4:5] = 0.7      # Belt overlapping
        
        # Compute loss
        loss = matrix_integrity_loss(pred_matrix)
        
        # Check if gradients flow
        loss.backward()
        
        # Gradients should be non-zero for cells with overlap
        self.assertFalse(torch.all(pred_matrix.grad[0, self.assembler_ch, 4, 4] == 0))
        self.assertFalse(torch.all(pred_matrix.grad[0, self.belt_ch, 4, 4] == 0))
        
    def test_realistic_blueprint_case(self):
        """Test with a more realistic blueprint-like configuration."""
        simple_bp_str = "0eJyVktFugzAMRf/Fz0kFFNrB435jmqZArS5ScKIkTEOIf58pHUKj3danKI59zpXiAWrTofOaIlQD6MZSgOplgKDPpMxUi71DqEBHbEEAqXa6qRCwrY2ms2xV864JZQqjAE0n/IQqHV8FIEUdNc683+YEOBu41dLk43GZ7goBPVTJrmDoSXts5ufkiu3fqGtr9KwSwK96zkgBfeTqKBbjUttYknuSw0aSrYDRKwrO+ihrNHGLfSD7foUNzuh4M2d2waV/4/IVruOP8Gdv+byTU2bXoHKDZuP3r5PrptkfpuIh039Etos3TYdpky67V61WVcAH+jAjntL8WGbHPCvLfcnbYBSn4O7npXscvwCGe/P8"
        f = Factory.from_str(simple_bp_str)
        pred_matrix = f.get_tensor(dims=(20,20))
        
        loss = matrix_integrity_loss(pred_matrix)
        
        # This is a valid configuration, so loss should be low
        self.assertLess(loss.item(), 0.001)
       
        # loss_with_overlap = matrix_integrity_loss(pred_matrix)
        
        # # Loss should increase significantly
        # self.assertGreater(loss_with_overlap.item(), loss.item() + 0.1)

if __name__ == "__main__":
    unittest.main()