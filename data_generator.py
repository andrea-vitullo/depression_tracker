from keras.utils import Sequence
import numpy as np
import h5py
import logging


class DataGenerator(Sequence):
    # noinspection PyMissingConstructor
    def __init__(self, h5_filepath, batch_size, audio_shape, verbose=False):
        self.h5_filepath = h5_filepath
        self.batch_size = batch_size
        self.audio_shape = audio_shape
        self.verbose = verbose
        with h5py.File(self.h5_filepath, 'r') as f:
            self.num_samples = len(f.keys())

    def __len__(self):
        return int(np.ceil(self.num_samples / self.batch_size))

    def __getitem__(self, idx):
        batch_x = []
        batch_y = []

        start = idx * self.batch_size
        end = min((idx + 1) * self.batch_size, self.num_samples)

        if self.verbose:
            logging.info(f"Processing batch {idx + 1}/{self.__len__()}, indexes {start} to {end}")

        with h5py.File(self.h5_filepath, 'r') as f:
            keys = list(f.keys())[start:end]
            for key in keys:
                group = f[key]
                audio = group['audio'][:]

                original_label = group.attrs['label']
                transformed_label = 0 if original_label in [0, 1] else 1

                # Error handling to catch issues with specific files
                try:
                    # Check if audio shape matches the expected shape when flattened
                    expected_size = np.prod(self.audio_shape)

                    if audio.size == expected_size:
                        audio = audio.reshape(self.audio_shape)
                        batch_x.append(audio)
                        batch_y.append(transformed_label)
                    else:
                        print(f"Skipping {key}: incompatible shape {audio.shape}, expected {self.audio_shape}")

                    if self.verbose:
                        logging.info(f"Processed {key}: shape {audio.shape}, label {transformed_label}")

                except Exception as e:
                    logging.error(f"Error processing {key}: {e}")

        result = {"input_1": np.array(batch_x)}, np.array(batch_y)

        return result
