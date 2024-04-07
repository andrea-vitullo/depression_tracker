import unittest
import numpy as np
import os
import h5py
from multi_data_loader import MultiDataLoader


class TestMultiDataLoader(unittest.TestCase):

    mock_h5_path = '/Users/andreavitullo/Desktop/audio_test_DAIC-WOZ_DATABASE/mock_data.h5'

    @classmethod
    def setUpClass(cls):
        # Create a mock H5 file for testing
        os.makedirs(os.path.dirname(cls.mock_h5_path), exist_ok=True)
        with h5py.File(cls.mock_h5_path, 'w') as f:
            for i in range(30):  # Creating 30 samples
                grp = f.create_group(str(i))
                grp.create_dataset('audio', data=np.random.rand(120, 13))
                grp.attrs['label'] = i % 4  # Mock labels [0, 1, 2, 3]

    @classmethod
    def tearDownClass(cls):
        # Cleanup the mock H5 file after tests
        os.remove(cls.mock_h5_path)

    def test_initialization(self):
        feature_shapes = {'mfcc': (120, 13)}
        data_loader = MultiDataLoader({'mfcc': self.mock_h5_path}, {'mfcc': self.mock_h5_path},
                                      {'mfcc': self.mock_h5_path}, feature_shapes)

        self.assertIsNotNone(data_loader.train_dataset)
        self.assertIsNotNone(data_loader.dev_dataset)
        self.assertIsNotNone(data_loader.test_dataset)

    def test_data_generator_integration(self):
        feature_shapes = {'mfcc': (120, 13)}
        data_loader = MultiDataLoader({'mfcc': self.mock_h5_path}, {'mfcc': self.mock_h5_path},
                                      {'mfcc': self.mock_h5_path}, feature_shapes)

        for dataset in [data_loader.train_dataset, data_loader.dev_dataset, data_loader.test_dataset]:
            for inputs, outputs in dataset.take(1):  # Taking one batch for the test
                self.assertEqual(inputs['mfcc'].shape[1:], (120, 13))
                self.assertTrue(len(outputs.numpy()) > 0)  # Check if outputs have been loaded


if __name__ == '__main__':
    unittest.main()
