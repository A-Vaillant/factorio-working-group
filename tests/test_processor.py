"""
Tests for processor.py functions and classes
"""
import warnings
warnings.filterwarnings("ignore", message="Unknown entity 'ee-infinity-loader'")

import json
import unittest
import random
from pathlib import Path
from unittest.mock import patch, mock_open, MagicMock
from copy import deepcopy

import numpy as np
from src.representation import Factory
from src.processor import EntityPuncher
from draftsman.blueprintable import Blueprint, BlueprintBook, get_blueprintable_from_string

# from src.blueprint_pipeline.utils import map_entity_to_channel
# from src.syntheticdata.processor import recursive_blueprint_book_parse, FactoryLoader, EntityPuncher
test_factory_str = '0eNqtmN9vmzAQx/+Vyc8wgX8AjrTX/QHbYxVVDnFbq2AzIOuqKP/7TGhW1CK4M3uKZezPfX32+Xw5k0N10k1rbE92Z2J6XZPdpC8ilTroyvf9+Pn9S5JQ32NKZzuyuzuTzjxaVQ0Traq1H6S6TteHytjHuFblk7E6TsnFT7FH/Yfs0ks0M6lvle0a1/axt9RPhtPLPiK/ddsZZ8lOZFRyKf0PkyxjEdG2N73Ro5Q3lrGdbnvdepmN6/znYeaZeFgsiq8iIq++lfvW5W3+67091Qc/YRQ3L2kRJq+wo2l1OY7gn9B0gtY6NvbBWP89rpw6zmuVi/j+tRlQ7tQ3p0Hdg6n8mkdH3Fwd/XNJ62z8qFUbvzxpv5UR+XVSlTfvv1nX1mroKl3dqFb1zqsl38ig97YJ20D7m7r72h0HysuTP2SV6QbdH9zEJm7qPK6KdeVX3Zoyblyl5/yULe8pR+1pjtlTgUJnGHQ2QS8daLG8+BylUGAUFig0x6Al/hCwlcBOUGoZRm2aAveKrmikKI0UpZGh2CmKjYuwBMUWMN9yueLbDKORS5RGVJBxVNZIC3Qo8HzFFRIlF3Uh0gS4XSt3NkUlYo66WSkqzDjqTqTTMJt9CC0+I7KrBc8319yuTr2r1TA27kqjbanjRpXPn5Mm5QFmxXazIsAs3W42w5udRHSw2TzAbLbdbAHMLu/HiH88qMVnqtzyzuJzMcuAsT95Ys1zUvTTB7BgRoFUvqKObXmbzCM5+v0AWTAwc05y/by6DJ2BIepyILVYUbcpP84jJTqHARbMoZlRLKvjoSUqW89efEuNyrbWqKVrGt3GTaV6/f/rSs4Cy0CI33hgHQhhi8AKDsLOAks4CDsPLLgg7CKwUIKwZWChBGCLJLBQgrDTwAIHwqaBBQ6EzQKrEQibB1YRELYIrCIg7Ax7D3MOuYeNHa/hj+by4SZ98TOGm/mORyJKk0jso2uTjc2hJ0rle5vxsS2H9tjPJv2M3/r30/9+/wK6sTWg'

class TestFactoryBasics(unittest.TestCase):
    def setUp(self):
        self.f = Factory.from_str(test_factory_str)

    def test_blueprint_entities_name(self):
        for e in self.f.blueprint.entities:
            self.assertIsNotNone(getattr(e, 'name'))

    def test_blueprint_to_dict_name(self):
        for e in self.f.blueprint.to_dict()['blueprint']['entities']:
            self.assertIn('name', e)

    def test_basic_factory_functionality(self):
        for e in self.f.json['blueprint']['entities']:
            self.assertIn('name', e)

    def test_deepcopying(self):
        bp = deepcopy(self.f.blueprint)
        for e in bp.to_dict()['blueprint']['entities']:
            self.assertIn('name', e)


class TestEntityPuncher(unittest.TestCase):
    def setUp(self):
        self.f = Factory.from_str(test_factory_str)
        self.ep = EntityPuncher(self.f)

    def test_blueprint_entities_name(self):
        for e in self.f.blueprint.entities:
            self.assertIsNotNone(getattr(e, 'name'))

    def test_blueprint_to_dict_name(self):
        for e in self.f.blueprint.to_dict()['blueprint']['entities']:
            self.assertIn('name', e)

    def test_factory_json_entities(self):
        for e in self.f.blueprint.entities:
            self.assertIsNotNone(getattr(e, 'name'))

    def test_entitypuncher_origfactory_bp_entities(self):
        for e in self.ep.original_factory.blueprint.entities:
            self.assertIsNotNone(getattr(e, 'name'))

    def test_entitypuncher_origfactory_bp_json(self):
        for e in self.ep.original_factory.blueprint.to_dict()['blueprint']['entities']:
            self.assertIn('name', e)

    def test_entitypuncher_origfactory_json_entities(self):
        # This fails, for some ungodly reason.
        for e in self.ep.original_factory.json['blueprint']['entities']:
            self.assertIn('name', e)

    def test_entitypuncher_factory_json_entities(self):
        for e in self.ep.blueprint.to_dict()['blueprint']['entities']:
            self.assertIn('name', e)

    def test_hack_to_solve_factory_functionality(self):
        self.f.json = self.f.blueprint.to_dict()
        for e in self.f.json['blueprint']['entities']:
            self.assertIn('name', e)

    def test_removability(self):
        random_entity = random.choice(self.f.blueprint.entities)
        self.ep.blueprint.entities.remove(random_entity)
        J = self.ep.blueprint.to_dict()
        for e in J['blueprint']['entities']:
            self.assertIn('name', e)

    def test_punchsequence(self):
        before, after, cs = self.ep.generate_state_action_pairs()
        for b, a, c in zip(before, after, cs):
            self.assertEqual(len(b.blueprint.entities)-1, len(a.blueprint.entities))
            self.assertIn('name', a.json['blueprint']['entities'][0])
            self.assertIn('name', b.json['blueprint']['entities'][0])
        

if __name__ == '__main__':
    unittest.main()