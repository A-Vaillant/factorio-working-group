""" utils.py
Various utilities used to get information about blueprints.
"""
import math
from draftsman.blueprintable import Blueprint, Blueprintable, get_blueprintable_from_JSON, get_blueprintable_from_string
from draftsman.utils import string_to_JSON
from dataclasses import dataclass
from typing import Optional
import numpy as np
import math
from draftsman.blueprintable import Blueprint, Blueprintable, get_blueprintable_from_JSON, get_blueprintable_from_string
from draftsman.utils import string_to_JSON
from dataclasses import dataclass
from typing import Optional
from functools import lru_cache

import logging

# You surely will not regret putting a global variable in your representation module.
REPR_VERSION = 4

from draftsman.data import recipes as recipe_data
from draftsman.data import items



def recursive_json_parse(json_bp: dict) -> list[dict]:
    if 'blueprint' in json_bp:
        # it's a blueprint!
        return [json_bp]
    # it's not.
    elif not 'blueprint_book' in json_bp:
        # It's something ELSE. SKIP IT.
        return []
    blueprints = []
    for bp in json_bp['blueprint_book']['blueprints']:
        blueprints += recursive_json_parse(bp)
    return blueprints

def recursive_blueprint_book_parse(bp_book: Blueprintable) -> list[Blueprint]:
    # Reached a leaf.
    if isinstance(bp_book, Blueprint):
        return [bp_book]

    blueprints = []
    for bp_node in bp_book.blueprints:
        blueprints += recursive_blueprint_book_parse(bp_node)
    return blueprints

def get_item_id_from_name(item_name: str) -> int:
    # Note: Wooden Chests have an index of 0, so we have to offset everything.
    if item_name == 'ee-infinity-loader':
        item_name = 'underground-belt'
    arr_index = [i for i in items.raw.keys()].index(item_name)
    return arr_index + 1

def get_item_name_from_id(item_id: int) -> str:
    arr_name = [i for i in items.raw.keys()][item_id - 1]
    return arr_name

def _first_product_name(recipe_name: str) -> str | None:
    """Return the name of the first product (if any) of a recipe."""
    r = recipe_data.raw.get(recipe_name, {})
    if "products" in r:             # Factorio ≥0.18 style
        return r["products"][0]["name"]
    if "results" in r:              # Some mods / prototypes
        return r["results"][0]["name"]
    return r.get("result")          # Legacy single‑result field

def get_entity_topleft(entity):
    # Uses Draftsman to get some data, returns topleft coords.
    from draftsman.data import entities as entity_data
    pos = entity['position']
    if entity['name'] == 'ee-infinity-loader':  # Custom modded sink/source object.
        (l, t), (r, b) = [-0.4, -0.4], [0.4, 0.4]
    else:
        raw_entity = entity_data.raw.get(entity['name'])
        (l, t), (r, b) = raw_entity['selection_box']
        if raw_entity is None:
            return
    return (round(pos['x']+l),
            round(pos['y']+t))

def get_entity_bottomright(entity):
    # Uses Draftsman to get some data, returns topleft coords.
    from draftsman.data import entities as entity_data
    pos = entity['position']
    raw_entity = entity_data.raw.get(entity['name'])
    if raw_entity is None:
        return
    (l, t), (r, b) = raw_entity['selection_box']
    return (round(pos['x']+r),
            round(pos['y']+b))

def bound_bp_by_entities(bbook: Blueprint):
    left = 100
    top = 100
    for m in bbook.entities:
        left = min(left, m.tile_position._data[0])
        top = min(top, m.tile_position._data[1])
    return left, top

def bound_bp_by_json(jsf: dict):
    """Returns the top-left most point that defines the minimal
    rectangle. In a compacted matrix, this returns (0,0).

    Args:
        jsf (dict): The JSON representation of a Factory.
    """
    left = 100
    top = 100
    for m in jsf['blueprint']['entities']:
        a, b = get_entity_topleft(m)
        left = min(left, a)
        top = min(top, b)
    return left, top

def map_entity_to_key(entity) -> str:
    key = None
    # Determine which matrix to use based on entity type (using string comparison)
    if entity.type == "assembling-machine":
        key = "assembler"
    elif entity.type == "inserter":
        key = "inserter"
    elif entity.type in ["transport-belt", "splitter", "underground-belt"]:
        key = "belt"
    elif entity.type in ["electric-pole"]:
        key = "pole"
    else:
        pass
    return key