import numpy as np
import librosa
from sklearn.preprocessing import StandardScaler
import warnings

import my_config
from utils import audio_utils


def extract_raw_audio(audio, sr, target_length=my_config.NSEG * my_config.H):
    # If the audio's length is more than target_length, we truncate it
    if len(audio) > target_length:
        audio = audio[:target_length]
    # If the audio's length is less than target_length, we pad it with zeros
    elif len(audio) < target_length:
        audio = np.concatenate([audio, np.zeros(target_length - len(audio))])

    # Calculate mean and std
    mean = np.mean(audio)
    std = np.std(audio)

    # Standardize audio data
    audio = audio_utils.standardization(audio, mean, std)

    print(audio.shape)

    return audio[:, np.newaxis]


def extract_mfcc_segments(audio, sr, n_mels=40, n_fft=1024, hop_length=512, n_mfcc=13):
    # The length of each segment in samples to fit exactly 120 frames
    segment_length_samples = 61440  # 61,440 samples to fit the requirement

    # Calculate the total number of segments that can be extracted from the audio
    num_segments = len(audio) // segment_length_samples

    mfcc_segments = []

    for i in range(num_segments):
        # Calculate the start and end sample indices for the current segment
        start_sample = i * segment_length_samples
        end_sample = start_sample + segment_length_samples

        # Extract the segment from the audio
        segment = audio[start_sample:end_sample]

        # Compute the MFCCs for the current segment
        mfccs = librosa.feature.mfcc(segment, sr=sr, n_mfcc=n_mfcc, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)

        # Transpose the MFCCs matrix
        mfccs_transposed = mfccs.T

        # # Calculate mean and std for each segment for normalization
        # mean = np.mean(mfccs_transposed)
        # std = np.std(mfccs_transposed)
        #
        # # Normalize the mfccs for the segment
        # mfccs_normalized = (mfccs_transposed - mean) / (std + 1e-8)

        # Optional: Subtract the mean of each coefficient from all frames (mean normalization per bin)
        mfccs_transposed -= np.mean(mfccs_transposed, axis=0, keepdims=True)

        # Trim the last time frame if mfcc shape exceeds the expected frame count
        if mfccs_transposed.shape[0] > 120:
            mfccs_transposed = mfccs_transposed[:120, :]

        # Append the processed, normalized MFCC segment to the list
        mfcc_segments.append(mfccs_transposed[:120])

    # Optionally, you can check the shape of one segment to verify the dimensions
    if mfcc_segments:
        print(f"One MFCC segment shape: {mfcc_segments[0].shape} (Expected: (120, {n_mfcc}))")

    return mfcc_segments


def compute_global_mel_stats(file_paths):
    all_mels = []
    total_files = len(file_paths)
    print(f"Starting computation of global Mel stats for {total_files} files.")

    for i, file_path in enumerate(file_paths):
        print(f"Processing file {i + 1}/{total_files}: {file_path}")
        try:
            audio, sr = librosa.load(file_path, sr=None)  # Load with the original sampling rate
            melspectrogram = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=my_config.N_MELS,
                                                            n_fft=my_config.MEL_N_FFT,
                                                            hop_length=my_config.MEL_HOP_LENGTH)
            logmelspec = librosa.power_to_db(melspectrogram)
            all_mels.extend(logmelspec.flatten())  # Flatten and extend the list of melspectrograms
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue  # Skip this file and move to the next

    # Compute the global mean and standard deviation
    global_mel_mean = np.mean(all_mels)
    global_mel_std = np.std(all_mels)

    print(f"Global Mel mean: {global_mel_mean}, Global Mel std: {global_mel_std}")

    # Make sure to return these values
    return global_mel_mean, global_mel_std


def extract_logmel(audio, sr, n_mels=my_config.N_MELS, length=my_config.MEL_LENGTH,
                   hop_length=my_config.MEL_HOP_LENGTH, n_fft=my_config.MEL_HOP_LENGTH,
                   mean=None, std=None):

    # Compute log-mel spectrogram features
    melspectrogram = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels,
                                                    n_fft=n_fft, hop_length=hop_length)
    logmelspec = librosa.power_to_db(melspectrogram)

    # Transpose to shape (timesteps, features) to align with our expected model input shape
    logmelspec_transposed = logmelspec.T

    # Normalize using pre-computed global mean and std
    if mean is not None and std is not None:
        logmelspec_normalized = (logmelspec_transposed - mean) / std
    else:
        raise ValueError("Global Mel mean and std have not been computed.")

    # Initialize a zero array with the target shape (L, D)
    logmel_padded = np.zeros((length, n_mels))

    # If the log-mel spectrogram is shorter than L frames, pad it with zeros
    if logmelspec_normalized.shape[0] < length:
        logmel_padded[:logmelspec_normalized.shape[0], :] = logmelspec_normalized
    else:  # Truncate if it's longer
        logmel_padded = logmelspec_normalized[:length, :]

    # Add an additional dimension to mimic a single-channel image for Conv2D
    logmel_padded = logmel_padded[..., np.newaxis]

    print(logmel_padded.shape)

    return logmel_padded


# APPLIED NORMALISATION BASED ON LOUDER TRAIN AUDIO ##################################################################

# def extract_logmel_segments(audio, sr, n_mels=40, n_fft=512, hop_length=512, mean=None, std=None):
#     # The length of each segment in samples to fit exactly 120 frames
#     segment_length_samples = 61440  # 61,440 samples to fit the requirement
#
#     # Calculate the total number of segments that can be extracted from the audio
#     num_segments = len(audio) // segment_length_samples
#
#     logmel_segments = []
#
#     for i in range(num_segments):
#         # Calculate the start and end sample indices for the current segment
#         start_sample = i * segment_length_samples
#         end_sample = start_sample + segment_length_samples
#
#         # Extract the segment from the audio
#         segment = audio[start_sample:end_sample]
#
#         # Compute the mel spectrogram for the current segment
#         melspectrogram = librosa.feature.melspectrogram(segment, sr=sr, n_mels=n_mels, n_fft=n_fft,
#                                                         hop_length=hop_length)
#
#         # Convert the power spectrogram to decibel units
#         logmelspec = librosa.power_to_db(melspectrogram)
#
#         logmelspec_transposed = logmelspec.T
#
#         # Normalize the log-mel spectrogram if mean and std are provided
#         if mean is not None and std is not None:
#             logmelspec = (logmelspec_transposed - mean) / std
#
#         # Optional: Subtract the mean of each mel frequency bin from all frames (mean normalization per bin)
#         logmelspec -= np.mean(logmelspec, axis=1, keepdims=True)
#
#         # Trim the last time frame if the spectrogram shape exceeds the expected frame count
#         if logmelspec.shape[0] > 120:
#             logmelspec = logmelspec[:120: ,]
#
#         # Append the processed log-mel spectrogram segment to the list
#         logmel_segments.append(logmelspec)
#
#     # Optionally, you could print the shape of one segment to verify the dimensions
#     if logmel_segments:
#         print(f"One log-mel segment shape: {logmel_segments[0].shape} (Expected: (120, 40))")
#
#     return logmel_segments


# APPLIED Z-NORMALISATION ON EACH SEGMENT ############################################################################


def extract_logmel_segments(audio, sr, n_mels=40, n_fft=1024, hop_length=512):
    # The length of each segment in samples to fit exactly 120 frames
    segment_length_samples = 61440  # 61,440 samples to fit the requirement

    # Calculate the total number of segments that can be extracted from the audio
    num_segments = len(audio) // segment_length_samples

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

        # Calculate mean and std for each segment for normalization
        # mean = np.mean(logmelspec_transposed)
        # std = np.std(logmelspec_transposed)
        #
        # # Normalize the log-mel spectrogram for the segment
        # logmelspec_normalized = (logmelspec_transposed - mean) / (std + 1e-8)  # 1e-8 is just a very small number and it's often called epsilon in machine learning context.

        # Optional: Subtract the mean of each mel frequency bin from all frames (mean normalization per bin)
        # This step is done after segment-wise normalization to ensure each bin's relative level is maintained
        logmelspec_transposed -= np.mean(logmelspec_transposed, axis=1, keepdims=True)

        # Trim the last time frame if the spectrogram shape exceeds the expected frame count
        if logmelspec_transposed.shape[0] > 120:
            logmelspec_transposed = logmelspec_transposed[:120, :]

        # Append the processed, normalized log-mel spectrogram segment to the list
        logmel_segments.append(logmelspec_transposed)

    # Optionally, you could print the shape of one segment to verify the dimensions
    if logmel_segments:
        print(f"One log-mel segment shape: {logmel_segments[0].shape} (Expected: (120, 40))")

    return logmel_segments


# SPECTROGRAM ON EACH SEGMENT #######################################################################################


def extract_spectrogram_segments(audio, sr, n_fft=1024, hop_length=512):
    # The length of each segment in samples to fit exactly 120 frames
    segment_length_samples = 61440  # 61,440 samples to fit the requirement

    # Calculate the total number of segments that can be extracted from the audio
    num_segments = len(audio) // segment_length_samples

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

        # Calculate mean and std for each segment for normalization
        # mean = np.mean(logspectrogram_transposed)
        # std = np.std(logspectrogram_transposed)

        # Normalize the log-spectrogram for the segment
        #logspectrogram_normalized = (logspectrogram_transposed - mean) / (std + 1e-8)  # 1e-8 prevents division by zero

        # Optional: Subtract the mean of each frequency bin from all frames (mean normalization per bin)
        # This step is done after segment-wise normalization to ensure each bin's relative level is maintained
        logspectrogram_transposed -= np.mean(logspectrogram_transposed, axis=0, keepdims=True)

        if logspectrogram_transposed.shape[0] > 120:
            logspectrogram_transposed = logspectrogram_transposed[:120, :]

        # Append the processed, normalized log-spectrogram segment to the list
        spectrogram_segments.append(logspectrogram_transposed)

    # Optionally, you could print the shape of one segment to verify the dimensions
    if spectrogram_segments:
        # n_fft//2 + 1 is the number of unique STFT bins
        print(f"One spectrogram segment shape: {spectrogram_segments[0].shape} (Expected: (120, {n_fft // 2 + 1}))")

    return spectrogram_segments


def extract_chroma_segments(audio, sr, n_fft=1024, hop_length=512):
    # The length of each segment in samples to fit exactly 120 frames
    segment_length_samples = 61440  # 61,440 samples to fit the requirement

    # Calculate the total number of segments that can be extracted from the audio
    num_segments = len(audio) // segment_length_samples

    chroma_segments = []

    for i in range(num_segments):
        # Calculate the start and end sample indices for the current segment
        start_sample = i * segment_length_samples
        end_sample = start_sample + segment_length_samples

        # Extract the segment from the audio
        segment = audio[start_sample:end_sample]

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=np.ComplexWarning)
            chroma = librosa.feature.chroma_stft(segment, sr=sr, n_fft=n_fft, hop_length=hop_length)

            # Transpose the chroma matrix
            chroma_transposed = chroma.T

            # # Calculate mean and std for each segment for normalization
            # mean = np.mean(chroma_transposed)
            # std = np.std(chroma_transposed)
            #
            # # Normalize the chroma features for the segment
            # chroma_normalized = (chroma_transposed - mean) / (std + 1e-8)

            # Optional: Subtract the mean of each coefficient from all frames (mean normalization per bin)
            chroma_transposed -= np.mean(chroma_transposed, axis=0, keepdims=True)

            # Trim the last time frame if chroma shape exceeds the expected frame count
            if chroma_transposed.shape[0] > 120:
                chroma_transposed = chroma_transposed[:120, :]

            # Append the processed, normalized chroma segment to the list
            chroma_segments.append(chroma_transposed)

    # Optionally, you can check the shape of one segment to verify the dimensions
    if chroma_segments:
        print(f"One chroma segment shape: {chroma_segments[0].shape} (Expected: (120, 12))")  # 12 chroma features

    return chroma_segments
