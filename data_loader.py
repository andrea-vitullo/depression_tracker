import tensorflow as tf

from my_config import BATCH_SIZE, LOGMEL_SHAPE_WINDOW
from data_generator import DataGenerator


class DataLoader:
    def __init__(self, train_file, dev_file, test_file):
        self.train_generator = DataGenerator(train_file, BATCH_SIZE, LOGMEL_SHAPE_WINDOW, verbose=False)
        self.dev_generator = DataGenerator(dev_file, BATCH_SIZE, LOGMEL_SHAPE_WINDOW, verbose=False)
        self.test_generator = DataGenerator(test_file, BATCH_SIZE, LOGMEL_SHAPE_WINDOW, verbose=False)

        self.train_dataset = self._create_dataset(self.train_generator)
        self.dev_dataset = self._create_dataset(self.dev_generator)

    @staticmethod
    def _create_dataset(generator):
        def gen():
            for features, labels in generator:
                yield features, labels

        dataset = tf.data.Dataset.from_generator(
            gen,
            output_signature=(
                {"input_1": tf.TensorSpec(shape=(None,) + LOGMEL_SHAPE_WINDOW, dtype=tf.float32)},
                tf.TensorSpec(shape=(None,), dtype=tf.float32),
            ),
        )
        return dataset
