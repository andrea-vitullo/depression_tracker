import unittest

import my_config
from utils import utils
from collections import Counter


class LoadFilesLabelsTest(unittest.TestCase):

    def setUp(self):
        self.AUDIO_TRAIN_DIRS = [my_config.AUDIO_TRAIN_DIR_0 + '/male',
                                 my_config.AUDIO_TRAIN_DIR_0 + '/female',
                                 my_config.AUDIO_TRAIN_DIR_1 + '/male',
                                 my_config.AUDIO_TRAIN_DIR_1 + '/female']

        self.AUDIO_DEV_DIRS = [my_config.AUDIO_DEV_DIR_0 + '/male',
                               my_config.AUDIO_DEV_DIR_0 + '/female',
                               my_config.AUDIO_DEV_DIR_1 + '/male',
                               my_config.AUDIO_DEV_DIR_1 + '/female']

        self.AUDIO_TEST_DIRS = [my_config.AUDIO_TEST_DIR_0 + '/male',
                                my_config.AUDIO_TEST_DIR_0 + '/female',
                                my_config.AUDIO_TEST_DIR_1 + '/male',
                                my_config.AUDIO_TEST_DIR_1 + '/female']

        # Load the data
        self.train_files, self.train_labels = utils.load_files_labels(self.AUDIO_TRAIN_DIRS, my_config.LABELS)
        self.dev_files, self.dev_labels = utils.load_files_labels(self.AUDIO_DEV_DIRS, my_config.LABELS)
        self.test_files, self.test_labels = utils.load_files_labels(self.AUDIO_TEST_DIRS, my_config.LABELS)

    def test_number_files_vs_labels(self):
        # Check that the number of files matches the number of labels
        self.assertEqual(len(self.train_files), len(self.train_labels),
                         "Mismatch between number of files and labels in training data.")
        self.assertEqual(len(self.dev_files), len(self.dev_labels),
                         "Mismatch between number of files and labels in development data.")

    def test_label_counts(self):
        train_counter = Counter(self.train_labels)
        dev_counter = Counter(self.dev_labels)
        print(train_counter)
        print(dev_counter)
        # Add assertions here to check the contents of train_counter and dev_counter if necessary


if __name__ == '__main__':
    unittest.main()
