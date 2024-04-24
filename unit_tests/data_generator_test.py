import unittest
import numpy as np
import h5py
import os
from data_management.data_generator import DataGenerator


class TestMultiDataGenerator(unittest.TestCase):

    mock_h5_paths = {
        'mfcc': '/Users/andreavitullo/Desktop/audio_test_DAIC-WOZ_DATABASE/mock_mfcc.h5',
        'chroma': '/Users/andreavitullo/Desktop/audio_test_DAIC-WOZ_DATABASE/mock_chroma.h5',
        'logmel': '/Users/andreavitullo/Desktop/audio_test_DAIC-WOZ_DATABASE/mock_logmel.h5',
        'spectrogram': '/Users/andreavitullo/Desktop/audio_test_DAIC-WOZ_DATABASE/mock_spectrogram.h5'
    }
    feature_shapes = {
        'mfcc': (120, 13),
        'chroma': (120, 128),
        'logmel': (120, 256),
        'spectrogram': (120, 513)  # Adjust these shapes as per your actual data
    }

    @classmethod
    def setUpClass(cls):
        for feature, path in cls.mock_h5_paths.items():
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with h5py.File(path, 'w') as f:
                for i in range(20):  # Create 20 samples to cover all label scenarios
                    grp = f.create_group(str(i))
                    data_shape = (120,) + (cls.feature_shapes[feature][1],)
                    grp.create_dataset('audio', data=np.random.rand(*data_shape))
                    grp.attrs['label'] = i % 4  # Labels will be 0, 1, 2, and 3

    @classmethod
    def tearDownClass(cls):
        for path in cls.mock_h5_paths.values():
            if os.path.exists(path):
                os.remove(path)

    def test_feature_shapes(self):
        batch_size = 5
        generator = DataGenerator(self.mock_h5_paths, batch_size, self.feature_shapes)

        inputs, outputs = next(iter(generator))

        for feature, expected_shape in self.feature_shapes.items():
            expected_shape = (batch_size,) + expected_shape
            self.assertEqual(inputs[feature].shape, expected_shape,
                             f"Shape mismatch for {feature}")

    def test_batch_shape(self):
        batch_size = 10
        generator = DataGenerator(self.mock_h5_paths, batch_size, self.feature_shapes)

        inputs, _ = next(iter(generator))

        for feature, expected_shape in self.feature_shapes.items():
            expected_shape = (batch_size,) + expected_shape
            self.assertEqual(inputs[feature].shape, expected_shape,
                             f"{feature} shape mismatch")

    def test_label_transformation(self):
        """Test that labels are correctly transformed from [0, 1, 2, 3] to binary [0, 1]."""
        batch_size = 10
        for feature, path in self.mock_h5_paths.items():
            mock_files = {feature: path}
            generator = DataGenerator(mock_files, batch_size, {feature: self.feature_shapes[feature]})
            _, outputs = next(iter(generator))
            # Assuming the binary transformation logic is already applied in your DataGenerator
            unique_labels = np.unique(outputs)
            expected_labels = [0, 1]  # Expected binary labels
            for label in unique_labels:
                self.assertIn(label, expected_labels, f"{feature} labels not correctly transformed to binary format")

    def test_handling_incorrect_paths(self):
        """Testing how the generator handles incorrect paths for each feature."""
        batch_size = 10
        for feature in self.feature_shapes.keys():
            mock_files = {feature: 'path/to/nonexistent_file.h5'}
            with self.assertRaises(Exception, msg=f"{feature} incorrect path not handled"):
                generator = DataGenerator(mock_files, batch_size, {feature: self.feature_shapes[feature]})
                _ = next(iter(generator))  # Attempt to generate the first batch


if __name__ == '__main__':
    unittest.main()
