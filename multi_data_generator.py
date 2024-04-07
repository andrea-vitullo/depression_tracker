from collections import defaultdict
from keras.utils import Sequence
import numpy as np
import h5py


class MultiDataGenerator(Sequence):

    def __init__(self, h5_files, batch_size, feature_shapes, verbose=False):
        self.h5_files = h5_files
        self.batch_size = batch_size
        self.feature_shapes = feature_shapes  # Store the feature shapes
        self.verbose = verbose

    def __len__(self):
        # Here we assume every h5 file contains the same number of samples
        with h5py.File(next(iter(self.h5_files.values())), 'r') as f:
            self.num_samples = len(f.keys())
        return int(np.ceil(self.num_samples / self.batch_size))

    def __getitem__(self, idx):
        batch_x = defaultdict(list)
        batch_y = []

        start = idx * self.batch_size
        end = min((idx + 1) * self.batch_size, self.num_samples)

        # Only need to get labels once per batch, not on each feature iteration
        with h5py.File(next(iter(self.h5_files.values())), 'r') as f:
            keys = list(f.keys())[start:end]
            for key in keys:
                group = f[key]
                # Original label extraction is common for all features
                original_label = group.attrs['label']
                transformed_label = 0 if original_label in [0, 1] else 1
                batch_y.append(transformed_label)

        # Now, serialize the features separately
        for feature_type, h5_filepath in self.h5_files.items():
            with h5py.File(h5_filepath, 'r') as f:
                for key in keys:
                    group = f[key]
                    audio = group['audio'][:]
                    # Append to the corresponding feature_type
                    batch_x[feature_type].append(audio)

        inputs = {name: np.array(data) for name, data in batch_x.items()}
        outputs = np.array(batch_y)

        return inputs, outputs
