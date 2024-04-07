import tensorflow as tf
import os

from my_config import BATCH_SIZE, FEATURE_SHAPES
from multi_data_generator import MultiDataGenerator


class MultiDataLoader:

    def __init__(self, train_files, dev_files, test_files, feature_shapes):
        self.train_files = train_files
        self.dev_files = dev_files
        self.test_files = test_files
        self.feature_shapes = feature_shapes  # Store the feature shapes

        self.train_dataset = self.load_dataset(self.train_files)

        print("First batch check:")
        for inputs, targets in self.train_dataset.take(1):
            for name, input_tensor in inputs.items():
                print(f"{name} shape: {input_tensor.shape}")
            print(f"Targets shape: {targets.shape}")

        self.dev_dataset = self.load_dataset(self.dev_files)
        self.test_dataset = self.load_dataset(self.test_files)

    def load_dataset(self, file_dict):
        generator = MultiDataGenerator(file_dict, BATCH_SIZE, self.feature_shapes)

        def gen():
            for features, labels in generator:
                yield features, labels

        dataset = tf.data.Dataset.from_generator(
            gen,
            output_signature=(
                {key: tf.TensorSpec(shape=(None,) + FEATURE_SHAPES[key][0],
                                    dtype=tf.float32) for key in file_dict.keys()},
                tf.TensorSpec(shape=(None,), dtype=tf.float32),
            ),
        )

        return dataset

    @staticmethod
    def check_files_exist(file_dict):
        """Checks if all files in the provided dictionary exist."""
        for file_path in file_dict.values():
            if not os.path.isfile(file_path):
                raise FileNotFoundError(f"The file {file_path} does not exist.")
