from rng_utils import _rng
from models import Item


def build_item_bank(n_items: int = 90, seed: int = None):
    rng = _rng(seed)
    items = []
    families = ["lsystem", "shape", "colored", "groovy", "alpha_perlin"]
    for i in range(n_items):
        family = rng.choice(families)
        b = float(rng.uniform(-2.0, 2.0))
        a = float(rng.uniform(0.7, 1.7))
        base_seed = int(rng.randint(0, 1_000_000))
        items.append(Item(id=i, family=family, a=a, b=b, base_seed=base_seed))
    return items
