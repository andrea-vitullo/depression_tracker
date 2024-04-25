import numpy as np
import librosa

import my_config
from utils import utils
import features_extractors


# EXTRACTION FUNCTION
# based on extraction type to perform import from features_extractors
EXTRACTION_FUNCTIONS = {
    'mfcc': features_extractors.extract_mfcc_segments,
    'chroma': features_extractors.extract_chroma_segments,
    'logmel': features_extractors.extract_logmel_segments,
    'spectrogram': features_extractors.extract_spectrogram_segments
}


# Load the data
train_files, train_labels = utils.load_files_labels(my_config.AUDIO_TRAIN_DIR)
dev_files, dev_labels = utils.load_files_labels(my_config.AUDIO_DEV_DIR)
test_files, test_labels = utils.load_files_labels(my_config.AUDIO_TEST_DIR)


def find_min_segments_per_file(files, extraction_functions, sr=22050):
    """
    Determine the minimum number of segments that can be extracted from any audio file in the dataset,
    across all specified features.

    Args:
        files (list): List of paths to audio files.
        extraction_functions (dict): Dictionary mapping feature names to their respective extraction functions.
                                     Each extraction function should accept an audio signal and sampling rate as
                                     input, and return the total number of segments that can be extracted when
                                     called with count_segments=True.
        sr (int, optional): Sampling rate to use when loading audio files. Defaults to 22050.

    Returns:
        int: Minimum number of segments that can be extracted from any audio file.
    """

    minimum_segments = None

    segments_list = []

    for file_path in files:
        audio, _ = librosa.load(file_path, sr=sr)

        # For each feature type, determine the number of segments that can be extracted
        for feature_name, extraction_func in extraction_functions.items():
            num_segments = extraction_func(audio, sr, count_segments=True)
            segments_list.append(num_segments)

            if minimum_segments is None or num_segments < minimum_segments:
                minimum_segments = num_segments

    return minimum_segments, segments_list


segment_counts_sample = []


try:
    min_segments, segment_counts_sample = find_min_segments_per_file(train_files, EXTRACTION_FUNCTIONS)
    print(f"Minimum number of segments per file across all features: {min_segments}")
except Exception as e:
    print(f"Error while calculating minimum segments: {e}")


# Calculates the percentiles for the sample list
percentiles = [25, 50, 75]
for p in percentiles:
    value = np.percentile(segment_counts_sample, p)
    print(f"{p}th percentile: {value}")
