import numpy as np
from dataclasses import dataclass

@dataclass
class Shadow:
    global_best_position: np.ndarray
    coords: tuple
    shadow_level: float
    coefficient: int