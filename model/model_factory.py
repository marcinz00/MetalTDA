import abc
from abc import abstractmethod

from tensorflow import keras
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.optimizer_v2.adam import Adam

from config import Configuration


class ModelFactory(abc.ABC):
    def create_model(
            self,
            config: Configuration
    ) -> Model:
        feature_extractor = self.get_feature_extractor(config)

        model = keras.models.Sequential()
        model.add(feature_extractor)
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(config.number_of_neurons, activation='relu'))
        model.add(keras.layers.Dropout(config.dropout_rate))
        model.add(keras.layers.Dense(config.output_neurons, activation='softmax'))

        trainable_layers = self.get_trainable_layers()
        for layer in feature_extractor.layers:
            layer.trainable = layer.name in trainable_layers

        model.compile(
            loss=config.loss_function,
            optimizer=Adam(lr=config.learning_rate),
            metrics=['acc']
        )

        return model

    @abstractmethod
    def get_feature_extractor(self, config):
        pass

    def get_trainable_layers(self):
        return {}
