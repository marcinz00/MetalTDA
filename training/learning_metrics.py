from dataclasses import dataclass
from typing import Any, Dict

from numpy import ndarray


@dataclass
class LearningMetrics:
    history: Any
    confusion_matrix: ndarray
    classification: Dict
    classes: Dict
