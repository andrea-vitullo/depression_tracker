import numpy as np
import librosa
from sklearn.preprocessing import StandardScaler
import warnings

import my_config
from utils import audio_utils


def extract_raw_audio_segments(audio, sr, target_length=61440, mean=0, std=1):

    # Calculate the total number of segments that can be extracted from the audio
    num_segments = len(audio) // target_length

    raw_audio_segments = []

    for i in range(num_segments):
        # Calculate the start and end sample indices for the current segment
        start_sample = i * target_length
        end_sample = start_sample + target_length

        # Extract the segment from the audio
        segment = audio[start_sample:end_sample]

        # Standardize audio data
        segment = audio_utils.standardization(segment, mean, std)

        raw_audio_segments.append(segment[:, np.newaxis])

    # Optionally, you could print the shape of one segment to verify the dimensions
    if raw_audio_segments:
        print(f"One raw audio segment shape: {raw_audio_segments[0].shape} (Expected: (61440, 1))")

    return raw_audio_segments


def extract_mfcc_segments(audio, sr, n_mels=256, n_fft=1024, hop_length=128, n_mfcc=13, mean=0, std=1, count_segments=False):
    # The length of each segment in samples to fit exactly 120 frames
    segment_length_samples = 245760  # 61,440 samples to fit the requirement   122,880
    # Calculate the total number of segments that can be extracted from the audio
    num_segments = len(audio) // segment_length_samples

    print(num_segments)

    if count_segments:
        return num_segments

    mfcc_segments = []

    for i in range(num_segments):
        # Calculate the start and end sample indices for the current segment
        start_sample = i * segment_length_samples
        end_sample = start_sample + segment_length_samples

        # Extract the segment from the audio
        segment = audio[start_sample:end_sample]

        # Check if the segment is too short (this check should technically never trigger with fixed segment sizes, but it's good to have for debugging)
        if len(segment) < n_fft:
            print(f"Segment too short: {len(segment)} samples, expected at least {n_fft}")
            continue  # Skip this segment

        # Debugging output before the MFCC computation
        # print(
        #     f"Processing segment: {i + 1}/{num_segments}, Segment length: {len(segment)}, Sample Rate: {sr}, n_mfcc: {n_mfcc}, n_mels: {n_mels}, n_fft: {n_fft}, hop_length: {hop_length}")

        # Compute the MFCCs for the current segment
        mfccs = librosa.feature.mfcc(segment, sr=sr, n_mfcc=n_mfcc, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)

        # Transpose the MFCCs matrix
        mfccs_transposed = mfccs.T

        # # Normalize the mfccs for the segment
        # mfccs_normalized = (mfccs_transposed - mean) / (std + 1e-8)
        #
        # # Optional: Subtract the mean of each coefficient from all frames (mean normalization per bin)
        # mfccs_normalized -= np.mean(mfccs_normalized, axis=0, keepdims=True)

        # Trim the last time frame if mfcc shape exceeds the expected frame count
        if mfccs_transposed.shape[0] > 480:
            mfccs_transposed = mfccs_transposed[:480, :]
        elif mfccs_transposed.shape[0] < 480:
            pad_width = ((0, 480 - mfccs_transposed.shape[0]), (0, 0))  # Padding applies only to time axis
            mfccs_transposed = np.pad(mfccs_transposed, pad_width, mode='constant')

        # Append the processed, normalized MFCC segment to the list
        mfcc_segments.append(mfccs_transposed)

    # Optionally, you can check the shape of one segment to verify the dimensions
    if mfcc_segments:
        print(f"One MFCC segment shape: {mfcc_segments[0].shape} (Expected: (120, {n_mfcc}))")

    return mfcc_segments


def compute_global_stats(files, extraction_func):
    # Initialize count, mean, M2
    count = 0
    mean = 0.0
    M2 = 0.0

    total_files = len(files)
    print(f"Starting computation of global stats for {total_files} files.")

    # For each file...
    for i, file_path in enumerate(files):
        print(f"Processing file {i + 1}/{total_files}: {file_path}")
        try:
            # Load audio file
            audio, sr = librosa.load(file_path, sr=None)

            # Extract segments
            segments = extraction_func(audio, sr)

            # Flatten segments and iterate over each value
            for value in np.concatenate(segments).ravel():
                # Increment count
                count += 1

                # Update mean and M2
                delta = value - mean
                mean += delta / count
                delta2 = value - mean
                M2 += delta * delta2

        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    # Compute standard deviation
    variance_n = M2 / (count if count > 0 else 1)
    std_dev = np.sqrt(variance_n)

    print(f"Global mean: {mean}, Global std: {std_dev}")

    return mean, std_dev


def extract_logmel_segments(audio, sr, n_mels=256, n_fft=1024, hop_length=512, mean=0, std=1, count_segments=False):
    # The length of each segment in samples to fit exactly 120 frames
    segment_length_samples = 245760  # 61,440 samples to fit the requirement

    # Calculate the total number of segments that can be extracted from the audio
    num_segments = len(audio) // segment_length_samples

    # print(num_segments)
    #
    # if count_segments:
    #     return num_segments

    logmel_segments = []

    for i in range(num_segments):
        # Calculate the start and end sample indices for the current segment
        start_sample = i * segment_length_samples
        end_sample = start_sample + segment_length_samples

        # Extract the segment from the audio
        segment = audio[start_sample:end_sample]

        # Compute the mel spectrogram for the current segment
        melspectrogram = librosa.feature.melspectrogram(segment, sr=sr, n_mels=n_mels, n_fft=n_fft,
                                                        hop_length=hop_length)

        # Convert the power spectrogram to decibel units
        logmelspec = librosa.power_to_db(melspectrogram)

        logmelspec_transposed = logmelspec.T

        # # Calculate mean and std for each segment for normalization
        # mean = np.mean(logmelspec_transposed)
        # std = np.std(logmelspec_transposed)

        # Normalize the log-mel spectrogram for the segment
        # logmelspec_normalized = (logmelspec_transposed - mean) / (std + 1e-8)  # 1e-8 is just a very small number and it's often called epsilon in machine learning context.
        #
        # # Optional: Subtract the mean of each mel frequency bin from all frames (mean normalization per bin)
        # # This step is done after segment-wise normalization to ensure each bin's relative level is maintained
        # logmelspec_normalized -= np.mean(logmelspec_normalized, axis=1, keepdims=True)

        # Trim the last time frame if the spectrogram shape exceeds the expected frame count
        if logmelspec_transposed.shape[0] > 480:
            logmelspec_transposed = logmelspec_transposed[:480, :]

        # Append the processed, normalized log-mel spectrogram segment to the list
        logmel_segments.append(logmelspec_transposed)

    # Optionally, you could print the shape of one segment to verify the dimensions
    if logmel_segments:
        print(f"One log-mel segment shape: {logmel_segments[0].shape} (Expected: (120, 256))")

    return logmel_segments


# SPECTROGRAM ON EACH SEGMENT #######################################################################################


def extract_spectrogram_segments(audio, sr, n_fft=1024, hop_length=512, mean=0, std=1, count_segments=False):
    # The length of each segment in samples to fit exactly 120 frames
    segment_length_samples = 245760  # 61,440 samples to fit the requirement

    # Calculate the total number of segments that can be extracted from the audio
    num_segments = len(audio) // segment_length_samples

    # print(num_segments)
    #
    # if count_segments:
    #     return num_segments

    spectrogram_segments = []

    for i in range(num_segments):
        # Calculate the start and end sample indices for the current segment
        start_sample = i * segment_length_samples
        end_sample = start_sample + segment_length_samples

        # Extract the segment from the audio
        segment = audio[start_sample:end_sample]

        # Compute the STFT for the current segment
        stft = librosa.stft(segment, n_fft=n_fft, hop_length=hop_length)

        # Compute the spectrogram, which is the squared magnitude of each component in STFT matrix
        spectrogram = np.abs(stft)**2

        # Convert the power spectrogram to decibel units
        logspectrogram = librosa.power_to_db(spectrogram)

        logspectrogram_transposed = logspectrogram.T

        # # Calculate mean and std for each segment for normalization
        # mean = np.mean(logspectrogram_transposed)
        # std = np.std(logspectrogram_transposed)

        # Normalize the log-spectrogram for the segment
        logspectrogram_normalized = (logspectrogram_transposed - mean) / (std + 1e-8)  # 1e-8 prevents division by zero

        # Optional: Subtract the mean of each frequency bin from all frames (mean normalization per bin)
        # This step is done after segment-wise normalization to ensure each bin's relative level is maintained
        logspectrogram_normalized -= np.mean(logspectrogram_normalized, axis=0, keepdims=True)

        if logspectrogram_normalized.shape[0] > 120:
            logspectrogram_normalized = logspectrogram_normalized[:120, :]

        # Append the processed, normalized log-spectrogram segment to the list
        spectrogram_segments.append(logspectrogram_normalized)

    # Optionally, you could print the shape of one segment to verify the dimensions
    if spectrogram_segments:
        # n_fft//2 + 1 is the number of unique STFT bins
        print(f"One spectrogram segment shape: {spectrogram_segments[0].shape} (Expected: (120, {n_fft // 2 + 1}))")

    return spectrogram_segments


def extract_chroma_segments(audio, sr, n_fft=1024, hop_length=128, mean=0, std=1, count_segments=False):
    # The length of each segment in samples to fit exactly 120 frames
    segment_length_samples = 245760  # 61,440 samples to fit the requirement, 122880

    # Calculate the total number of segments that can be extracted from the audio
    num_segments = len(audio) // segment_length_samples

    # print(num_segments)
    #
    # if count_segments:
    #     return num_segments

    chroma_segments = []

    for i in range(num_segments):
        # Calculate the start and end sample indices for the current segment
        start_sample = i * segment_length_samples
        end_sample = start_sample + segment_length_samples

        # Extract the segment from the audio
        segment = audio[start_sample:end_sample]

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=np.ComplexWarning)
            chroma = librosa.feature.chroma_stft(segment, sr=sr, n_fft=n_fft, hop_length=hop_length, n_chroma=12)

            # Transpose the chroma matrix
            chroma_transposed = chroma.T

            # # Calculate mean and std for each segment for normalization
            # mean = np.mean(chroma_transposed)
            # std = np.std(chroma_transposed)

            # # Normalize the chroma features for the segment
            # chroma_normalized = (chroma_transposed - mean) / (std + 1e-8)
            #
            # # Optional: Subtract the mean of each coefficient from all frames (mean normalization per bin)
            # chroma_normalized -= np.mean(chroma_normalized, axis=0, keepdims=True)

            # Trim the last time frame if chroma shape exceeds the expected frame count
            if chroma_transposed.shape[0] > 480:
                chroma_transposed = chroma_transposed[:480, :]
            elif chroma_transposed.shape[0] < 480:
                pad_width = ((0, 480 - chroma_transposed.shape[0]), (0, 0))  # Padding applies only to time axis
                chroma_transposed = np.pad(chroma_transposed, pad_width, mode='constant')

            # Append the processed, normalized chroma segment to the list
            chroma_segments.append(chroma_transposed)

    # Optionally, you can check the shape of one segment to verify the dimensions
    if chroma_segments:
        print(f"One chroma segment shape: {chroma_segments[0].shape} (Expected: (120, 12))")  # 12 chroma features

    return chroma_segments


# def compute_global_stats(files, extraction_func):
#     all_values = []
#     total_files = len(files)
#     print(f"Starting computation of global stats for {total_files} files.")
#
#     for i, file_path in enumerate(files):
#         print(f"Processing file {i+1}/{total_files}: {file_path}")
#         try:
#             audio, sr = librosa.load(file_path, sr=None)  # Load with the original sampling rate
#             segments = extraction_func(audio, sr)
#             # Flatten and extend the list of all segments
#             all_values.extend(np.concatenate(segments).ravel())
#         except Exception as e:
#             print(f"Error processing {file_path}: {e}")
#             continue  # Skip this file and move to the next
#
#     mean = np.mean(all_values)
#     std = np.std(all_values)
#
#     print(f"Global mean: {mean}, Global std: {std}")
#
#     return mean, std
