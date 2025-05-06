import unittest
import numpy as np
from unittest.mock import MagicMock, patch
from draftsman.blueprintable import get_blueprintable_from_string
import io
import sys
import json

# Import the modules to test
from representation import (
    map_entity_to_key, Factory, recursive_json_parse, 
    recursive_blueprint_book_parse, bound_bp_by_entities,
    RepresentationError, trim_zero_edges, center_in_N,
    REPR_VERSION
)


class TestTrimZeroEdges(unittest.TestCase):
    def test_trim_all_channels(self):
        # Example 1: 3-channel matrix with zero borders
        mat1 = np.zeros((5, 6, 3))
        mat1[1:4, 2:4, :] = 1  # Non-zero values in the middle

        # Trim using all channels (identity function)
        result1 = trim_zero_edges(mat1)
        self.assertEqual(result1.shape, (3, 2, 3))

    def test_single_channel_selection(self):
        # Example 2: Using only channel 0
        mat2 = np.zeros((5, 6, 3))
        mat2[1:4, 2:4, 0] = 1  # Non-zero values in channel 0
        mat2[1:3, 1:5, 1:3] = 1  # Different non-zero pattern in other channels

        # Trim considering only channel 0
        result2 = trim_zero_edges(mat2, lambda c: c == 0)
        self.assertEqual(result2.shape, (3, 2, 3))

    def test_2d_matrix(self):
        # Example 3: Single channel 2D matrix
        mat3 = np.zeros((4, 5))
        mat3[1:3, 1:4] = 1

        result3 = trim_zero_edges(mat3)
        self.assertEqual(result3.shape, (2, 3))

    def test_all_zeros_in_selected_channel(self):
        # Example 4: All zeros in selected channels
        mat4 = np.zeros((3, 4, 2))
        mat4[:, :, 1] = 1  # Only channel 1 has values

        # If we select only channel 0, everything is zero
        result4 = trim_zero_edges(mat4, lambda c: c == 0)
        self.assertEqual(result4.shape, (1, 1, 2))  # Minimal slice with both channels


class TestBlueprintMatrices(unittest.TestCase):
    def _test_blueprint(self, blueprint_string, expected_matrices):
        """Helper method to test a blueprint against expected matrices"""
        blueprint = get_blueprintable_from_string(blueprint_string)
        result_matrices = blueprint_to_opacity_matrices(blueprint)

        # Store for verbose output
        # self.blueprint_string = blueprint_string
        # self.result_matrices = result_matrices

        # Check that all expected keys exist
        self.assertEqual(result_matrices.shape, expected_matrices.shape)

        for i in range(expected_matrices.shape[2]):
            # print(result_matrices[:,:,i])
            # print(expected_matrices[:,:,i])
            self.assertTrue(np.all(result_matrices[:,:,i] == expected_matrices[:,:,i]))

    def test_simple_machine(self):
        """Test a simple machine blueprint"""
        blueprint_string = "0eJyd0s1qwzAMAOB30dketZuui497jTGGk4pOYCvBdsdCyLvPXsIohLDQm3+kT7LRCI27YR+IE5gRqO04gnkbIdKVrStnaegRDFBCDwLY+rJLwXLsu5Bkgy7BJID4gt9g1PQuADlRIpylJcPGiL5xxFfpbftJjFJlr+9iDu24VMrp8vB0EjDMi6xeKGA73x8Wd/jgm28wlFpi3GhoBevF3cHqB9g97R4fcdX/bnXnEkcMKZ+uRLXZqV6Jp12i3H78mnwuc/E7Q+Zu5AR8YYhzyouqzrU+V7quj7US4Gz+mhz9+hc9TT9q1eB9"
        h = 6
        w = 4
        c = 4
        expected_arr = [
                [
                    [0, 0, 0, 0],
                    [1, 1, 1, 0],
                    [1, 1, 1, 0],
                    [1, 1, 1, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                ],            
                [
                    [0, 1, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 0, 0],
                ],
                [
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [1, 1, 1, 0],
                ],
                [
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                ]
        ]
        expected = np.zeros((h, w, c))
        for ix, matrix in enumerate(expected_arr):
            for jx, row in enumerate(matrix):
                expected[jx, :, ix] = row
        self._test_blueprint(blueprint_string, expected)

    def test_machine_with_poles(self):
        blueprint_string = "0eJydk91uhCAQhd9lrqFZXLf+XO5rNKZBd+JOgmgAmxrjuxeqaUyNrbt3MAzfOQwzI5Sqx86QdpCPQFWrLeRvI1iqtVQh5oYOIQdy2AADLZuwc0Zq27XG8RKVg4kB6Rt+Qi6mggFqR45wJi03pLXYlIp0zRtZ3UkjF57Xtdantjoo+ev89HJhMMwLT72RwWo+Py3c4V33TYkmaLFxx9AGHC3cA9joCewRu+dnuOJ/brzikrZonI9uiGLXabQhXg4R+f7jt8jXFdI2UimOymcbqnjXKvzD74EKJL4vkep72fYmNF1aPKrGH+iP9JdaUoSe/56PfDVODD7Q2LkcqYiTLEriKMvOmWCgpP92n339yZ6mL6cfKSI="
        h = 7
        w = 4
        c = 4
        
        expected = np.zeros((h,w,c), dtype=np.int8)
        
        a_pos = [(1, 0)]
        i_pos = [(0, 1), (4, 1)]
        b_pos = [(5, 0), (5, 1), (5, 2)]
        p_pos = [(0, 2), (4, 0)]
        for ix, yx_pos in enumerate([a_pos, i_pos, b_pos, p_pos]):
            if yx_pos == a_pos:
                W = 3
                H = 3
            else:
                W = 1
                H = 1
            for (y, x) in yx_pos:
                expected[y:y+H, x:x+W, ix] = 1

        self._test_blueprint(blueprint_string, expected)


class TestMapEntityToKey(unittest.TestCase):
    """Test the map_entity_to_key function that maps entity types to matrix channels."""
    
    def test_entity_type_mapping(self):
        """Test mapping different entity types to the correct keys."""
        # Create a mock entity class
        class MockEntity:
            def __init__(self, entity_type):
                self.type = entity_type
        
        # Test assembling machine mapping
        assembler = MockEntity("assembling-machine")
        self.assertEqual(map_entity_to_key(assembler), "assembler")
        
        # Test inserter mapping
        inserter = MockEntity("inserter")
        self.assertEqual(map_entity_to_key(inserter), "inserter")
        
        # Test belt mappings (includes multiple types)
        belt = MockEntity("transport-belt")
        self.assertEqual(map_entity_to_key(belt), "belt")
        
        splitter = MockEntity("splitter")
        self.assertEqual(map_entity_to_key(splitter), "belt")
        
        underground = MockEntity("underground-belt")
        self.assertEqual(map_entity_to_key(underground), "belt")
        
        # Test pole mapping
        pole = MockEntity("electric-pole")
        self.assertEqual(map_entity_to_key(pole), "pole")
        
        # Test unknown entity type returns None
        unknown = MockEntity("unknown-type")
        self.assertIsNone(map_entity_to_key(unknown))


class TestRecursiveJsonParse(unittest.TestCase):
    """Test the recursive_json_parse function."""
    
    def test_single_blueprint(self):
        """Test with a single blueprint."""
        single_bp = {'blueprint': {'entities': []}}
        result = recursive_json_parse(single_bp)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], single_bp)
    
    def test_blueprint_book(self):
        """Test with a blueprint book containing multiple blueprints."""
        bp_book = {
            'blueprint_book': {
                'blueprints': [
                    {'blueprint': {'entities': []}},
                    {'blueprint': {'entities': []}}
                ]
            }
        }
        result = recursive_json_parse(bp_book)
        self.assertEqual(len(result), 2)
    
    def test_nested_blueprint_book(self):
        """Test with nested blueprint books."""
        nested_bp_book = {
            'blueprint_book': {
                'blueprints': [
                    {'blueprint': {'entities': []}},
                    {
                        'blueprint_book': {
                            'blueprints': [
                                {'blueprint': {'entities': []}},
                                {'blueprint': {'entities': []}}
                            ]
                        }
                    }
                ]
            }
        }
        result = recursive_json_parse(nested_bp_book)
        self.assertEqual(len(result), 3)
    
    def test_non_blueprint(self):
        """Test with non-blueprint and non-blueprint_book."""
        non_bp = {'something_else': {}}
        result = recursive_json_parse(non_bp)
        self.assertEqual(len(result), 0)


class TestBoundBpByEntities(unittest.TestCase):
    """Test the bound_bp_by_entities function."""
    
    def test_finding_bounds(self):
        """Test finding the bounds of entities in a blueprint."""
        # Create mock classes
        class MockPosition:
            def __init__(self, x, y):
                self._data = [x, y]
        
        class MockEntity:
            def __init__(self, x, y):
                self.tile_position = MockPosition(x, y)
        
        class MockBlueprint:
            def __init__(self, entities):
                self.entities = entities
        
        # Create a blueprint with entities at various positions
        entities = [
            MockEntity(5, 10),
            MockEntity(3, 8),
            MockEntity(7, 12)
        ]
        
        blueprint = MockBlueprint(entities)
        
        # Get bounds
        left, top = bound_bp_by_entities(blueprint)
        
        # Verify the bounds are correct (should find minimum x and y)
        self.assertEqual(left, 3)
        self.assertEqual(top, 8)
    
    def test_empty_blueprint(self):
        """Test with an empty blueprint (no entities)."""
        class MockBlueprint:
            def __init__(self):
                self.entities = []
        
        blueprint = MockBlueprint()
        
        # Default values should be returned
        left, top = bound_bp_by_entities(blueprint)
        self.assertEqual(left, 100)
        self.assertEqual(top, 100)


class TestFactoryClass(unittest.TestCase):
    """Tests for the Factory class."""
    
    @patch('src.representation.string_to_JSON')
    def test_from_str_success(self, mock_string_to_json):
        """Test creating a Factory from a valid blueprint string."""
        # Mock the string_to_JSON function
        mock_string_to_json.return_value = {'blueprint': {'entities': []}}
        
        # Test creating Factory from string
        factory = Factory.from_str("valid_blueprint_string")
        
        # Verify factory is created correctly
        self.assertIsNotNone(factory)
        self.assertIn('blueprint', factory.json)
    
    @patch('src.representation.string_to_JSON')
    def test_from_str_blueprint_book_error(self, mock_string_to_json):
        """Test that creating a Factory from a blueprint book string raises an error."""
        # Mock returning a blueprint book
        mock_string_to_json.return_value = {'blueprint-book': {}}
        
        # Should raise RepresentationError
        with self.assertRaises(RepresentationError):
            Factory.from_str("blueprint_book_string")
    
    @patch('src.representation.string_to_JSON')
    def test_from_str_unknown_error(self, mock_string_to_json):
        """Test that creating a Factory from an invalid string raises an error."""
        # Mock returning something invalid
        mock_string_to_json.return_value = {'not_a_blueprint': {}}
        
        # Should raise RepresentationError
        with self.assertRaises(RepresentationError):
            Factory.from_str("invalid_string")
    
    @patch('src.representation.get_blueprintable_from_string')
    @patch('src.representation.recursive_blueprint_book_parse')
    def test_from_blueprintable(self, mock_parse, mock_get_blueprintable):
        """Test creating Factories from a blueprintable."""
        # Create mock blueprints
        mock_bp1 = MagicMock()
        mock_bp2 = MagicMock()
        mock_bp1.to_dict.return_value = {'blueprint': {'entities': []}}
        mock_bp2.to_dict.return_value = {'blueprint': {'entities': []}}
        
        # Setup mocks
        mock_get_blueprintable.return_value = "blueprintable_object"
        mock_parse.return_value = [mock_bp1, mock_bp2]
        
        # Create factories from blueprintable
        factories = Factory.from_blueprintable("blueprintable_string")
        
        # Should return a list of Factory objects
        self.assertEqual(len(factories), 2)
        self.assertIsInstance(factories[0], Factory)
        self.assertIsInstance(factories[1], Factory)
    
    def test_blueprint_property(self):
        """Test the blueprint property loads the blueprint if needed."""
        # Create a factory with _bp=None
        factory = Factory(json={'blueprint': {'entities': []}})
        
        # Mock get_blueprintable_from_JSON
        with patch('src.representation.get_blueprintable_from_JSON') as mock_get:
            mock_blueprint = MagicMock()
            mock_get.return_value = mock_blueprint
            
            # Access blueprint property
            bp = factory.blueprint
            
            # Should call get_blueprintable_from_JSON
            mock_get.assert_called_once_with({'blueprint': {'entities': []}})
            self.assertEqual(bp, mock_blueprint)
    
    def test_get_matrix(self):
        """Test the get_matrix method with different representation versions."""
        # Create a Factory with a mock blueprint
        mock_blueprint = MagicMock()
        factory = Factory(json={'blueprint': {}}, _bp=mock_blueprint)
        
        # Mock blueprint_to_opacity_matrices
        with patch('src.representation.blueprint_to_opacity_matrices') as mock_matrix:
            mock_matrix.return_value = np.zeros((10, 10, 4))
            
            # Call get_matrix
            matrix = factory.get_matrix(dims=(10, 10), repr_version=1)
            
            # Should call blueprint_to_opacity_matrices
            mock_matrix.assert_called_once_with(mock_blueprint, 10, 10)
            self.assertEqual(matrix.shape, (10, 10, 4))


class TestErrorCases(unittest.TestCase):
    """Test various error cases and edge conditions."""
    
    def test_trim_zero_edges_all_zeros(self):
        """Test trim_zero_edges with a matrix of all zeros."""
        all_zeros = np.zeros((5, 5, 3))
        result = trim_zero_edges(all_zeros)
        # Should return minimal 1x1 slice
        self.assertEqual(result.shape, (1, 1, 3))
    
    def test_trim_zero_edges_no_channels_selected(self):
        """Test trim_zero_edges when no channels are selected."""
        matrix = np.ones((5, 5, 3))
        # Select no channels (always returns False)
        result = trim_zero_edges(matrix, channel_selector=lambda c: False)
        # Should return original matrix
        self.assertEqual(result.shape, (5, 5, 3))
    
    def test_center_in_N_large_matrix(self):
        """Test center_in_N with a matrix larger than N."""
        large_matrix = np.ones((20, 25))
        centered = center_in_N(large_matrix, N=15)
        # Result should be N×N (after transposing to C,H,W)
        self.assertEqual(centered.shape, (1, 15, 15))
        # Center should contain 1s (where the original matrix is placed)
        self.assertEqual(np.max(centered), 1)
    
    def test_center_in_N_small_matrix(self):
        """Test center_in_N with a matrix smaller than N."""
        small_matrix = np.ones((3, 4))
        centered = center_in_N(small_matrix, N=10)
        # Result should be N×N (after transposing to C,H,W)
        self.assertEqual(centered.shape, (1, 10, 10))
        # Center should contain 1s (where the original matrix is placed)
        self.assertEqual(np.max(centered), 1)
        # Corners should be 0
        self.assertEqual(centered[0, 0, 0], 0)
        self.assertEqual(centered[0, 9, 9], 0)
    
    def test_center_in_N_multichannel(self):
        """Test center_in_N with a multichannel matrix."""
        multichannel = np.ones((3, 4, 3))
        centered = center_in_N(multichannel, N=10)
        # Result should be C,N,N (after transposing)
        self.assertEqual(centered.shape, (3, 10, 10))


if __name__ == "__main__":
    unittest.main()