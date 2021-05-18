import logging
import math
import random
from dataclasses import dataclass
from typing import Tuple

from config import Configuration
from utils.path_provider import PathProvider
from utils.utils import make_dir, get_examples_by_category, copy_files_to_dir


@dataclass
class DataPreparer:
    TRAINING_DIR = 'training'
    TEST_DIR = 'test'

    configuration: Configuration
    data_path_provider: PathProvider
    new_path_provider: PathProvider

    def prepare_data(
            self
    ) -> Tuple[str, str, str]:
        temporary_dir = self.new_path_provider.provide()
        logging.info('Created temporary directory: %s', temporary_dir)
        training_dir = make_dir(temporary_dir, self.TRAINING_DIR)
        test_dir = make_dir(temporary_dir, self.TEST_DIR)

        examples_by_category = get_examples_by_category(self.data_path_provider.provide())
        test_size = math.ceil(
            min([len(examples) for examples in examples_by_category.values()]) * self.configuration.test_data)
        for category in examples_by_category:
            category_examples = examples_by_category[category].copy()
            random.shuffle(category_examples)

            test_examples = category_examples[:test_size]
            category_test_dir = make_dir(test_dir, category.name)
            copy_files_to_dir(test_examples, category_test_dir)

            training_examples = category_examples[test_size:]
            category_training_examples = make_dir(training_dir, category.name)
            copy_files_to_dir(training_examples, category_training_examples)

        return training_dir, test_dir, temporary_dir
