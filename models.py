from dataclasses import dataclass


@dataclass
class Item:
    id: int
    family: str  # "lsystem" | "shape" | "colored" | "groovy" | "alpha_perlin"
    a: float     # discrimination
    b: float     # difficulty
    base_seed: int  # seed to reproduce stimulus
