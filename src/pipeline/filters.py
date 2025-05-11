"""
filters.py

A bunch of callable classes that return bools.

Author: A. Vaillant (April 2025)
"""
from src.representation import get_entity_topleft, get_entity_bottomright, Factory


class Required:
    def __init__(self, entities):
        self.required = entities

    def __call__(self, factory: Factory) -> bool:
        try:
            entities = factory.json['blueprint']['entities']
        except KeyError:
            return False
        all_entities = set(e['name'] for e in entities)  # Everything has a name.
        return any(e in all_entities for e in self.required)  # Factory must have one of these.


class Whitelist:
    def __init__(self, entities):
        self.allows = entities

    def __call__(self, factory: Factory) -> bool:
        try:
            entities = factory.json['blueprint']['entities']
        except KeyError:
            return False
        all_entities = set(e['name'] for e in entities)  # Everything has a name.
        return all(e in self.allows for e in all_entities)  # All entities must be in the allowlist.


class RecipeWhitelist:
    def __init__(self, recipes):
        self.allows = recipes

    def __call__(self, factory: Factory) -> bool:
        try:
            entities = factory.json['blueprint']['entities']
        except KeyError:
            return False
        all_recipes = set(e.get('recipe') for e in entities)  # Will end up having None, so...
        all_recipes.remove(None)  # If a KeyError comes here, wrap it in a try/except.
        return all(r in self.allows for r in all_recipes)

class NoUnusedAssemblers:
    # Just kills any blueprint with unset assembler recipes.
    def __call__(self, factory: Factory) -> bool:
        try:
            entities = factory.json['blueprint']['entities']
        except KeyError:
            return False
        all_assemblers = [e for e in entities if e['name'].startswith('assembling')]
        return not any(a.get('recipe') is None for a in all_assemblers)

class Blacklist:
    def __init__(self, entities):
        self.bans = entities

    def __call__(self, factory: Factory) -> bool:
        try:
            entities = factory.json['blueprint']['entities']
        except KeyError:
            return False
        all_entities = set(e['name'] for e in entities)
        return all(e not in self.bans for e in all_entities)  # All entities must not be on the banlist.


class SizeRestrictor:
    def __init__(self, height, width):
        self.h = height
        self.w = width

    def __call__(self, factory: Factory) -> bool:
        left = 100
        top = 100
        right = -100
        bottom = -100

        for m in factory.json['blueprint']['entities']:
            a, b = get_entity_topleft(m)
            c, d = get_entity_bottomright(m)
            left = min(left, a)
            top = min(top, b)
            right = max(right, c)
            bottom = max(bottom, d)
        return ((right - left) <= self.h) and ((bottom - top) <= self.w)