import numpy as np
import h5py
import random


class FeatureAugmentor:
    def __init__(self, noise_level=0.005, time_mask_width=10, freq_mask_width=5):
        self.noise_level = noise_level
        self.time_mask_width = time_mask_width
        self.freq_mask_width = freq_mask_width

    def add_noise(self, data):
        noise = np.random.randn(*data.shape) * self.noise_level
        return data + noise

    def time_shift(self, data):
        shift = np.random.randint(-data.shape[0]//10, data.shape[0]//10)
        return np.roll(data, shift, axis=0)

    def time_mask(self, data):
        start = np.random.randint(0, data.shape[0] - self.time_mask_width)
        data[start:start + self.time_mask_width, :] = 0
        return data

    def frequency_mask(self, data):
        start = np.random.randint(0, data.shape[1] - self.freq_mask_width)
        data[:, start:start + self.freq_mask_width] = 0
        return data

    def augment(self, data):
        funcs = [self.add_noise, self.time_shift, self.time_mask, self.frequency_mask]
        random.shuffle(funcs)  # Apply augmentations in random order
        for func in funcs:
            data = func(data)
        return data


def apply_augmentation(h5_path, feature_key='audio'):
    """
    This function augments the feature data stored in the HDF5 file for each group.
    Augmented features are stored back into the HDF5 file with a new group named by adding '_aug' suffix to the
    original group name.

    Args:
        h5_path (str): Path to the HDF5 file storing the feature data.
        feature_key (str, optional): The key used to access feature data within each group in the HDF5 file, default
        is 'audio'.

    Returns:
          None

    Notes:
        - Augmentation is performed using the `FeatureAugmentor` class (assumed to pre-defined or imported).
        - Augmentation is applied to all groups found in the HDF5 file which do not end with '_aug'.
        - For each group, if augmented data already exists (determined by the presence of a group named
          original+'_aug'), the function skips augmentation for that group.
        - For new groups, the function creates a new group in the HDF5 file with the name as original+'_aug', and
          stores the augmented data as a new dataset in the group.
        - The group also duplicates the 'label' attribute from the original group.
    """

    augmentor = FeatureAugmentor()
    with h5py.File(h5_path, 'r+') as f:
        keys = [k for k in f.keys() if not k.endswith('_aug')]  # Only get original keys (keys without '_aug' suffix)
        for key in keys:
            augmented_key = f'{key}_aug'
            if augmented_key in f:
                print(f"Augmented data for group {key} already exists. Skipping augmentation.")
                continue
            print(f"Applying augmentation for group {key} ...")
            data = f[key][feature_key][:]
            augmented_data = augmentor.augment(data).astype(data.dtype)
            grp = f.create_group(augmented_key)
            grp.create_dataset(feature_key, data=augmented_data)
            grp.attrs['label'] = f[key].attrs['label']
            print(f"Augmentation for group {key} completed and saved as {augmented_key} in the HDF5 file.")
    print("All augmentations completed.")


apply_augmentation('/processed_audio_features/train_chroma.h5',
                   feature_key='audio')
