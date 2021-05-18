import logging
import os
from datetime import datetime

import tensorflow as tf
import yaml

from config import Configuration
from experiment import Experiment
from model.densenet_model_factory import DensenetModelFactory
from model.inception_model_factory import InceptionModelFactory
from model.inception_v3_model_factory import InceptionV3ModelFactory
from model.vgg16_model_factory import Vgg16ModelFactory
from model.vgg19_model_factory import Vgg19ModelFactory
from model.xception_model_factory import XceptionModelFactory
from runner.callbacks_factory import EarlyStoppingCallbacksFactory
from runner.data_generator import DataGenerator
from runner.data_preparer import DataPreparer
from runner.result_collector import ResultCollector
from runner.single_runner import SingleRunner
from utils.path_provider import SystemPathProvider, TemporaryPathProvider

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.8
tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.DEBUG)

with open("resources/properties.yaml", "r") as config_file:
    configuration: Configuration = yaml.load(config_file, Loader=yaml.Loader)

data_preparer = DataPreparer(
    configuration,
    SystemPathProvider(configuration.data_dir),
    TemporaryPathProvider()

)
data_generator = DataGenerator(
    configuration,
    data_preparer
)
callbacks_factory = EarlyStoppingCallbacksFactory()
timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

for experiment in [
    Experiment("densenet", DensenetModelFactory()),
    Experiment("inception", InceptionModelFactory()),
    Experiment("inceptionv3", InceptionV3ModelFactory()),
    Experiment("vgg16", Vgg16ModelFactory()),
    Experiment("vgg19", Vgg19ModelFactory()),
    Experiment("xception", XceptionModelFactory())
]:
    result_dir = os.path.join(
        configuration.result_dir,
        timestamp,
        experiment.directory
    )
    result_path_provider = SystemPathProvider(result_dir)
    runner = SingleRunner(data_generator, experiment.model_factory, callbacks_factory)
    result_collector = ResultCollector(result_path_provider)

    result_collector.collect_result(configuration, runner)
