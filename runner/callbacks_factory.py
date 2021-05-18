import abc
from abc import abstractmethod

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from config import Configuration


class CallbacksFactory(abc.ABC):
    @abstractmethod
    def create(
            self,
            configuration: Configuration
    ):
        pass


class EarlyStoppingCallbacksFactory(CallbacksFactory):
    def create(
            self,
            configuration: Configuration
    ):
        return [
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=configuration.reduce_learning_rate_factor,
                patience=configuration.reduce_learning_rate_patience,
                min_lr=configuration.min_learning_rate,
                cooldown=1,
                verbose=1
            ),
            EarlyStopping(
                monitor='val_loss',
                mode='min',
                restore_best_weights=True,
                verbose=1,
                patience=100
            )
        ]
