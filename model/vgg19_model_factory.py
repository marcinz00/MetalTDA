import tensorflow.keras as keras

from model.model_factory import ModelFactory


class Vgg19ModelFactory(ModelFactory):
    def get_feature_extractor(self, config):
        return keras.applications.VGG19(
            weights='imagenet',
            include_top=False,
            input_shape=config.input_shape
        )
