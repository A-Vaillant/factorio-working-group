"""
enrichment.py

A variety of functions that take ScrapedData and return additional fields.

Author: A. Vaillant (April 2025)
"""
from pathlib import Path
from draftsman.blueprintable import Blueprintable, Blueprint
import csv, json


# We want to just hand back strings to our intake classes, so write something
# that makes this doable.
# For each row, we pass its data to Draftsman, which returns a way to
# split the data across multiple JSON files.
def recursive_blueprint_book_parse(bp_book: Blueprintable) -> list[Blueprint]:
    # Reached a leaf.
    if isinstance(Blueprint, bp_book):
        return [bp_book]

    blueprints = []
    for bp_node in bp_book.blueprints:
        blueprints += recursive_blueprint_book_parse(bp_node)
    return blueprints


def quantify_entities(bp: Blueprint, granularity: int = 5) -> dict[str, int]:
    """Count entities in a blueprint with specified granularity."""
    # granularity: How specific we want. 5 will be everything, lower ones
    # will group together entities into classes (conveyor belts, assemblers)
    if granularity == 5:
        grain = lambda x: x
    else:
        # Add more granularity levels as needed
        grain = lambda x: x  

    # Create counter of entity names
    c = Counter(grain(entity.name) for entity in bp.entities)
    return dict(c)



# TODO: Write a map from every Factorio entity to the science tier.
tech_map = dict()
def quantify_tech_level(bp: Blueprint) -> int:
    # Using Factorio 1.0. 0 is initial, 1 is red science.
    tech_levels = {e: tech_map(e.name) for e in bp.entities}
    return max(tech_levels.values())
