import uuid
from dataclasses import dataclass

import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils import class_weight
from tensorflow.python.keras.backend import clear_session

from config import Configuration
from model.model_factory import ModelFactory
from runner.callbacks_factory import CallbacksFactory
from runner.data_generator import DataGenerator
from training.learning_metrics import LearningMetrics
from training.traning_result import TrainingResult


@dataclass
class SingleRunner:
    data_generator: DataGenerator
    model_factory: ModelFactory
    callbacks_factory: CallbacksFactory

    def run(
            self,
            configuration: Configuration,
    ) -> TrainingResult:
        clear_session()
        train_generator, test_generator, metrics_evaluator_generator = self.data_generator.get_generators()
        model = self.model_factory.create_model(configuration)
        model_id = str(uuid.uuid4())
        callbacks = self.callbacks_factory.create(configuration)

        class_weights_as_array = class_weight.compute_class_weight(
            class_weight='balanced',
            classes=np.unique(train_generator.classes),
            y=train_generator.classes)
        class_weights = dict(enumerate(class_weights_as_array))
        history = model.fit(
            train_generator,
            steps_per_epoch=np.ceil(train_generator.samples / train_generator.batch_size),
            epochs=configuration.epochs,
            validation_data=test_generator,
            validation_steps=np.ceil(test_generator.samples / test_generator.batch_size),
            verbose=2,
            callbacks=callbacks,
            class_weight=class_weights
        )

        best_model_prediction_result = model.predict(
            metrics_evaluator_generator,
            np.ceil(metrics_evaluator_generator.samples / metrics_evaluator_generator.batch_size)
        )
        model_prediction = np.argmax(best_model_prediction_result, axis=1)
        confusion = confusion_matrix(metrics_evaluator_generator.classes, model_prediction)
        classification = classification_report(metrics_evaluator_generator.classes, model_prediction, output_dict=True)

        return TrainingResult(
            model_id,
            model,
            history,
            LearningMetrics(
                history.history,
                confusion,
                classification,
                test_generator.class_indices
            )
        )
