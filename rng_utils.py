import numpy as np
from typing import Optional


def _rng(seed: Optional[int] = None) -> np.random.RandomState:
    return np.random.RandomState(seed) if seed is not None else np.random.RandomState()
