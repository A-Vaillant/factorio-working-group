import unittest
import numpy as np
from pathlib import Path
import io
import sys

from draftsman.blueprintable import get_blueprintable_from_string
from draftsman.blueprintable import Blueprint

from src.representation import blueprint_to_matrices
from src.representation import trim_zero_edges


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
        result_matrices = blueprint_to_matrices(blueprint)

        # Store for verbose output
        # self.blueprint_string = blueprint_string
        # self.result_matrices = result_matrices

        # Check that all expected keys exist
        self.assertEqual(set(result_matrices.keys()), set(expected_matrices.keys()))

        # Check each matrix
        for key, expected_matrix in expected_matrices.items():
            # Convert to numpy array if it's not already
            expected = np.array(expected_matrix)
            self.assertEqual(
                result_matrices[key].shape, expected.shape, f"Shape mismatch for {key}"
            )
            np.testing.assert_array_equal(
                result_matrices[key], expected, f"Values mismatch for {key}"
            )

    def test_simple_machine(self):
        """Test a simple machine blueprint"""
        blueprint_string = "0eJyd0s1qwzAMAOB30dketZuui497jTGGk4pOYCvBdsdCyLvPXsIohLDQm3+kT7LRCI27YR+IE5gRqO04gnkbIdKVrStnaegRDFBCDwLY+rJLwXLsu5Bkgy7BJID4gt9g1PQuADlRIpylJcPGiL5xxFfpbftJjFJlr+9iDu24VMrp8vB0EjDMi6xeKGA73x8Wd/jgm28wlFpi3GhoBevF3cHqB9g97R4fcdX/bnXnEkcMKZ+uRLXZqV6Jp12i3H78mnwuc/E7Q+Zu5AR8YYhzyouqzrU+V7quj7US4Gz+mhz9+hc9TT9q1eB9"
        expected = np.array([
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
        ])
        self._test_blueprint(blueprint_string, expected)

    def test_machine_with_poles(self):
        blueprint_string = "0eJydk91uhCAQhd9lrqFZXLf+XO5rNKZBd+JOgmgAmxrjuxeqaUyNrbt3MAzfOQwzI5Sqx86QdpCPQFWrLeRvI1iqtVQh5oYOIQdy2AADLZuwc0Zq27XG8RKVg4kB6Rt+Qi6mggFqR45wJi03pLXYlIp0zRtZ3UkjF57Xtdantjoo+ev89HJhMMwLT72RwWo+Py3c4V33TYkmaLFxx9AGHC3cA9joCewRu+dnuOJ/brzikrZonI9uiGLXabQhXg4R+f7jt8jXFdI2UimOymcbqnjXKvzD74EKJL4vkep72fYmNF1aPKrGH+iP9JdaUoSe/56PfDVODD7Q2LkcqYiTLEriKMvOmWCgpP92n339yZ6mL6cfKSI="
        h = 7
        w = 4
        c = 4
        
        expected = np.zeroes((h,w,c), dtype=np.int8)
        
        a_pos = [(1, 0)]
        
        for ix, yx_pos in [a_pos, i_pos, b_pos, p_pos]:
            if yx_pos == a_pos:
                W = 3
                H = 3
            else:
                W = 1
                H = 1
            for (y, x) in yx_pos:
                expected[y:y+H, x:x+W, ix] = 1
        expected = {
            "assembler": np.array(
                [
                    [0, 0, 0, 0],
                    [1, 1, 1, 0],
                    [1, 1, 1, 0],
                    [1, 1, 1, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                ],
                dtype=np.int8,
            ),
            "inserter": np.array(
                [
                    [0, 1, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                ],
                dtype=np.int8,
            ),
            "belt": np.array(
                [
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [1, 1, 1, 0],
                    [0, 0, 0, 0],
                ],
                dtype=np.int8,
            ),
            "pole": np.array(
                [
                    [0, 0, 1, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [1, 0, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                ],
                dtype=np.int8,
            ),
        }

        self._test_blueprint(blueprint_string, expected)


if __name__ == "__main__":
    unittest.main()
