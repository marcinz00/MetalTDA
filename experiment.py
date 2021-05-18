from dataclasses import dataclass

from model.model_factory import ModelFactory


@dataclass
class Experiment:
    directory: str
    model_factory: ModelFactory
