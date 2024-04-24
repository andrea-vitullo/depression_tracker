from collections import defaultdict
from keras.utils import Sequence
import numpy as np
import h5py


class DataGenerator(Sequence):
    def __init__(self, h5_files, batch_size, feature_shapes, use_augmented_data=False, verbose=False):
        self.h5_files = h5_files
        self.batch_size = batch_size
        self.feature_shapes = feature_shapes
        self.use_augmented_data = use_augmented_data
        self.verbose = verbose

    def __len__(self):
        with h5py.File(next(iter(self.h5_files.values())), 'r') as f:
            if self.use_augmented_data:
                self.num_samples = len(f.keys())
            else:  # if original data only
                self.num_samples = len([k for k in f.keys() if not k.endswith('_aug')])
        return int(np.ceil(self.num_samples / self.batch_size))

    def __getitem__(self, idx):
        batch_x = defaultdict(list)
        batch_y = []

        start = idx * self.batch_size
        end = min((idx + 1) * self.batch_size, self.num_samples)

        with h5py.File(next(iter(self.h5_files.values())), 'r') as f:
            if self.use_augmented_data:
                keys = list(f.keys())[start:end]
            else:  # if original data only
                keys = [k for k in f.keys() if not k.endswith('_aug')][start:end]

            for key in keys:
                group = f[key]
                original_label = group.attrs['label']
                transformed_label = 0 if original_label in [0, 1] else 1
                batch_y.append(transformed_label)

        for feature_type, h5_filepath in self.h5_files.items():
            with h5py.File(h5_filepath, 'r') as f:
                for key in keys:
                    group = f[key]
                    audio = group['audio'][:]
                    batch_x[feature_type].append(audio)

        inputs = {name: np.array(data) for name, data in batch_x.items()}
        outputs = np.array(batch_y)
        return inputs, outputs
