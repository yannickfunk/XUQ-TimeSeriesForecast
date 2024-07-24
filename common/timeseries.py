from dataclasses import dataclass
from typing import List, Union

import numpy as np


@dataclass
class TimeSeries:
    unique_id: str
    ds: Union[List[str], np.ndarray]
    y: np.ndarray
