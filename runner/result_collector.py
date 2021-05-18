import codecs
import json
import os
from dataclasses import dataclass
from typing import Generator

import jsonpickle

from config import Configuration
from runner.single_runner import SingleRunner
from training.learning_metrics import LearningMetrics
from training.traning_result import TrainingResult
from utils.path_provider import PathProvider
from utils.utils import make_dirs


@dataclass
class ResultCollector:
    resultPathProvider: PathProvider

    def collect_result(
            self,
            configuration: Configuration,
            runner: SingleRunner
    ):
        for result in self.run_multiple_models(configuration, runner):
            self.save_summary(result)

    @staticmethod
    def run_multiple_models(
            configuration: Configuration,
            runner: SingleRunner
    ) -> Generator[TrainingResult, None, None]:
        for i in range(configuration.number_of_models):
            yield runner.run(configuration)

    def save_summary(
            self,
            result: TrainingResult
    ):
        result_dir = self.resultPathProvider.provide()
        make_dirs(result_dir)
        model_id = result.model_id

        result.model.save(os.path.join(result_dir, f"{model_id}.pb"))
        self.save_history(os.path.join(result_dir, f"{model_id}.json"), result.metrics)

    @staticmethod
    def save_history(
            path: str,
            metrics: LearningMetrics):
        with codecs.open(path, 'w', encoding='utf-8') as file:
            encoded_metrics = jsonpickle.encode(metrics)
            json.dump(encoded_metrics, file, indent=4)
