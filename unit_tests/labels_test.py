import unittest
import os
from collections import Counter

import my_config
from utils import utils


class LoadFilesAndFeaturesTest(unittest.TestCase):

    def setUp(self):
        # Paths for loading wave files and their labels
        self.base_dirs = {
            'train': my_config.AUDIO_TRAIN_DIR,
            'dev': my_config.AUDIO_DEV_DIR,
            'test': my_config.AUDIO_TEST_DIR
        }

        # Paths for loading processed feature files
        self.feature_file_paths = {
            'train': my_config.TRAIN_H5,
            'dev': my_config.DEV_H5,
            'test': my_config.TEST_H5,
        }

    def test_load_files_labels(self):
        for dataset_type, base_dir in self.base_dirs.items():
            files, labels = utils.load_files_labels(base_dir)
            self.assertTrue(len(files) > 0, f"No files loaded for {dataset_type}.")
            self.assertEqual(len(files), len(labels),
                             f"Mismatch between number of files and labels in {dataset_type} data.")

    def test_load_files_labels_distribution(self):
        expected_distribution = {
            'train': {0: 35, 1: 25, 2: 16, 3: 16},
            'dev': {0: 10, 1: 11, 2: 5, 3: 7},
            'test': {0: 14, 1: 14, 2: 4, 3: 5}
        }

        tolerance = 0.10  # Adjust as needed

        for dataset_type, base_dir in self.base_dirs.items():
            _, labels = utils.load_files_labels(base_dir)
            label_counter = Counter(labels)

            for label, expected_count in expected_distribution[dataset_type].items():
                actual_count = label_counter[label]
                allowed_variance = expected_count * tolerance
                lower_bound = expected_count - allowed_variance
                upper_bound = expected_count + allowed_variance

                self.assertTrue(lower_bound <= actual_count <= upper_bound,
                                f"Label {label} count {actual_count} out of expected range [{lower_bound}, {upper_bound}] in {dataset_type}.")

    def test_load_features(self):
        for dataset_type, file_path in self.feature_file_paths.items():
            self.assertTrue(os.path.exists(file_path), f"Feature file does not exist for {dataset_type}.")
            audio_data, labels = utils.load_features(file_path)

            self.assertTrue(audio_data.shape[0] > 0, f"No audio data loaded for {dataset_type}.")
            self.assertEqual(audio_data.shape[0], labels.shape[0],
                             f"Mismatch between number of audio samples and labels in {dataset_type} features.")
            # Additional checks can be added here as needed


if __name__ == '__main__':
    unittest.main()
