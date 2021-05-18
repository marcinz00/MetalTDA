from dataclasses import dataclass
from typing import Any

from training.learning_metrics import LearningMetrics


@dataclass
class TrainingResult:
    model_id: str
    model: Any
    history: Any
    metrics: LearningMetrics
