"""
Tests for processor.py functions and classes
"""
import json
import unittest
from pathlib import Path
from unittest.mock import patch, mock_open, MagicMock

import numpy as np
from src.representation import Factory
from src.processor import EntityPuncher
from draftsman.blueprintable import Blueprint, BlueprintBook, get_blueprintable_from_string

# from src.blueprint_pipeline.utils import map_entity_to_channel
# from src.syntheticdata.processor import recursive_blueprint_book_parse, FactoryLoader, EntityPuncher


class TestEntityPuncher(unittest.TestCase):
    def setUp(self):
        with open('data/raw/txt/av/3machine.txt') as file:
            self.f = Factory.from_str(file.read())
    
    def test_punchsequence(self):
        ep = EntityPuncher(self.f)
        before, after, cs = ep.generate_state_action_pairs()
        for b, a, c in zip(before, after, cs):
            self.assertEqual(len(b.blueprint.entities)-1, len(a.blueprint.entities))
        

if __name__ == '__main__':
    unittest.main()