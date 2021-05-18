import tensorflow.keras as keras

from model.model_factory import ModelFactory


class XceptionModelFactory(ModelFactory):
    def get_feature_extractor(self, config):
        return keras.applications.Xception(
            weights='imagenet',
            include_top=False,
            input_shape=config.input_shape
        )
