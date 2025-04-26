"""
enrichment.py

A variety of functions that take ScrapedData and return additional fields.

Author: A. Vaillant (April 2025)
"""
import csv, json
from pathlib import Path
from collections import Counter


from draftsman.blueprintable import Blueprintable, Blueprint


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
