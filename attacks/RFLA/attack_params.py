from typing import List

import numpy as np
from dataclasses import dataclass


@dataclass
class AttackParams:
    points: List[np.array]
    red: int
    green: int
    blue: int
    alpha: float
    beta: float
    gamma: float
