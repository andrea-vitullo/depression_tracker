import h5py
import random

filename = './processed_audio_features/train_mfcc.h5'

with h5py.File(filename, 'r') as f:
    # List all groups
    print("Keys: %s" % f.keys())
    keys = list(f.keys())

    # Sample 10 random keys
    random_keys = random.sample(keys, 10)

    for key in random_keys:
        group = f[key]

        if isinstance(group, h5py.Group):
            print(f"Group name: {key}")
            print("Datasets in this group: %s" % group.keys())
            print(f"Attributes for group {key}: {dict(group.attrs)}")

            values = group['audio'][()]
            print(values)
        else:
            print(f"{key} is not a group.")
