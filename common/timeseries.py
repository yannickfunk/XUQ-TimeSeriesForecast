from dataclasses import dataclass
from typing import List, Optional, Union

import numpy as np


@dataclass
class TimeSeries:
    unique_id: str
    ds: Union[List[str], np.ndarray]
    y: np.ndarray


@dataclass
class AttributedTimeSeries:
    unique_id: str
    ds: Union[List[str], np.ndarray]
    y: np.ndarray
    positive_attributions: List[np.ndarray]
    negative_attributions: List[np.ndarray]
