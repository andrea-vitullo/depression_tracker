import h5py
import numpy as np
import os


train_files = {
    'mfcc': './processed_audio_features/train_mfcc.h5',
    'chroma': './processed_audio_features/train_chroma.h5',
    'logmel': './processed_audio_features/train_logmel.h5',
    'spectrogram': './processed_audio_features/train_spectrogram.h5',
}
dev_files = {
    'mfcc': './processed_audio_features/dev_mfcc.h5',
    'chroma': './processed_audio_features/dev_chroma.h5',
    'logmel': './processed_audio_features/dev_logmel.h5',
    'spectrogram': './processed_audio_features/dev_spectrogram.h5',
}
test_files = {
    'mfcc': './processed_audio_features/test_mfcc.h5',
    'chroma': './processed_audio_features/test_chroma.h5',
    'logmel': './processed_audio_features/test_logmel.h5',
    'spectrogram': './processed_audio_features/test_spectrogram.h5',
}


def balance_segments_in_dataset(h5_file_path):
    # Step 1: Calculate the minimum number of segments for any patient
    with h5py.File(h5_file_path, 'r') as h5f:
        patient_segments = {}
        for key in h5f.keys():
            patient_id = key.split('_')[0]
            if patient_id not in patient_segments:
                patient_segments[patient_id] = []
            patient_segments[patient_id].append(key)

        min_segments = min(len(segments) for segments in patient_segments.values())

    # Step 2: Create a new h5 file with balanced segments per patient
    temp_h5_file_path = h5_file_path.replace('.h5', '_balanced.h5')
    with h5py.File(h5_file_path, 'r') as input_h5, h5py.File(temp_h5_file_path, 'w') as output_h5:
        for patient_id, segments in patient_segments.items():
            if len(segments) > min_segments:
                selected_segments = np.random.choice(segments, min_segments, replace=False)
            else:
                selected_segments = segments

            for segment_id in selected_segments:
                features = input_h5[segment_id]['audio'][:]
                label = input_h5[segment_id].attrs['label']
                output_grp = output_h5.create_group(segment_id)
                output_grp.create_dataset('audio', data=features, compression='gzip')
                output_grp.attrs['label'] = label

    # Step 3: Replace the old h5 file with the new balanced one
    os.remove(h5_file_path)
    os.rename(temp_h5_file_path, h5_file_path)
    print(f"Dataset at {h5_file_path} has been balanced.")


# Call the function for each dataset
for dataset in [train_files, dev_files, test_files]:
    for feature in dataset:
        balance_segments_in_dataset(dataset[feature])

