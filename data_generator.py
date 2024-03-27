from keras.utils import Sequence
import numpy as np
import h5py
import gc


class DataGenerator(Sequence):
    # noinspection PyMissingConstructor
    def __init__(self, h5_filepath, batch_size):
        self.h5_filepath = h5_filepath
        self.batch_size = batch_size
        with h5py.File(self.h5_filepath, 'r') as f:
            self.num_samples = len(f.keys())

    def __len__(self):
        return int(np.ceil(self.num_samples / self.batch_size))

    def __getitem__(self, idx):
        batch_x = []
        batch_y = []

        start = idx * self.batch_size
        end = min((idx + 1) * self.batch_size, self.num_samples)

        with h5py.File(self.h5_filepath, 'r') as f:
            keys = list(f.keys())[start:end]
            for key in keys:
                group = f[key]
                audio = group['audio'][:]
                label = group.attrs['label']
                batch_x.append(audio)
                batch_y.append(label)

        result = {
            "input_1": np.array(batch_x)
        }, np.array(batch_y)

        gc.collect()

        return result
