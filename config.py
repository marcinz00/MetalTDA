from dataclasses import dataclass

from typing import Tuple


@dataclass
class ImageAugmentationConfiguration:
    rotation_range: int
    shear_range: float
    zoom_range: float
    horizontal_flip: bool
    fill_mode: str


@dataclass
class Configuration:
    batch_size: int
    epochs: int
    input_shape: Tuple[int, int, int]
    number_of_models: int
    test_data: float
    number_of_neurons: int
    loss_function: str
    dropout_rate: float
    output_neurons: int
    learning_rate: float
    min_learning_rate: float
    reduce_learning_rate_factor: float
    reduce_learning_rate_patience: int
    early_stopping_patience: int
    data_dir: str
    result_dir: str
    augmentation_configuration: ImageAugmentationConfiguration

    @property
    def target_size(self):
        return self.input_shape[:2]
