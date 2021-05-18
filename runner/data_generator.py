from dataclasses import dataclass

from keras_preprocessing.image import ImageDataGenerator

from config import Configuration
from runner.data_preparer import DataPreparer


@dataclass
class DataGenerator:
    configuration: Configuration
    data_preparer: DataPreparer

    def get_generators(
            self
    ):
        training_dir, test_dir, _ = self.data_preparer.prepare_data()
        augmentation_configuration = self.configuration.augmentation_configuration

        training_data_generator = ImageDataGenerator(
            rescale=1. / 255,
            rotation_range=augmentation_configuration.rotation_range,
            shear_range=augmentation_configuration.shear_range,
            zoom_range=augmentation_configuration.zoom_range,
            horizontal_flip=augmentation_configuration.horizontal_flip,
            fill_mode=augmentation_configuration.fill_mode
        )
        train_generator = training_data_generator.flow_from_directory(
            training_dir,
            target_size=self.configuration.target_size,
            batch_size=self.configuration.batch_size,
            class_mode='categorical'
        )

        test_data_generator = ImageDataGenerator(
            rescale=1. / 255
        )
        test_generator = test_data_generator.flow_from_directory(
            test_dir,
            target_size=self.configuration.target_size,
            batch_size=self.configuration.batch_size,
            class_mode='categorical'
        )

        metrics_evaluator_generator = test_data_generator.flow_from_directory(
            test_dir,
            target_size=self.configuration.target_size,
            batch_size=self.configuration.batch_size,
            class_mode='categorical',
            shuffle=False
        )

        return train_generator, test_generator, metrics_evaluator_generator
