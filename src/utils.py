import random

def generate_distinct_seeds(num_seeds: int, min_val: int = 0, max_val: int = 2**32 - 1) -> list[int]:
    """Generates a list of distinct random seeds."""
    if num_seeds > (max_val - min_val + 1):
        raise ValueError("Cannot generate more distinct seeds than the range allows.")
    
    seeds = set()
    while len(seeds) < num_seeds:
        seeds.add(random.randint(min_val, max_val))
    return list(seeds)