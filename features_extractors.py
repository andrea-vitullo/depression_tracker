import numpy as np
import librosa
from sklearn.preprocessing import StandardScaler

from my_config import *
from utils import audio_utils

def extract_raw_audio(audio, sr, target_length=NSEG * H):
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

    return audio


def extract_mfcc(audio, sr, d=N_MFCC, length=MAX_LENGTH_MFCC):
    # Compute MFCC features with the desired number of coefficients (D)
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=d, hop_length=MFCC_HOP_LENGTH, n_fft=N_FTT)

    # Transpose MFCCs to shape (time_steps, D) to align with Conv2D input expectations
    mfccs_transposed = mfccs.T

    # Scale features to have zero mean and unit variance
    scaler = StandardScaler()

    mfccs_scaled = scaler.fit_transform(mfccs_transposed)

    # Initialize a zero array with the target shape (L, D)
    mfcc_features_padded = np.zeros((length, d))

    # Copy the MFCCs into the padded array, truncating or leaving as-is based on length
    if mfccs_transposed.shape[0] > length:
        # If the actual number of time steps exceeds L, truncate
        mfcc_features_padded = mfccs_transposed[:length, :]
    else:
        # If fewer, pad the remaining with zeros
        mfcc_features_padded[:mfccs_transposed.shape[0], :] = mfccs_transposed

    # Add an additional dimension to mimic a single-channel image for Conv2D
    mfcc_features_padded = mfcc_features_padded[..., np.newaxis]

    print(mfcc_features_padded.shape)

    return mfcc_features_padded


def extract_logmel(audio, sr, n_mels=N_MELS, length=MEL_LENGTH, hop_length=MEL_HOP_LENGTH, n_fft=MEL_HOP_LENGTH):
    # Compute log-mel spectrogram features
    melspectrogram = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels,
                                                    n_fft=n_fft, hop_length=hop_length)
    logmelspec = librosa.power_to_db(melspectrogram)

    # Transpose to shape (timesteps, features) to align with our expected model input shape
    logmelspec_transposed = logmelspec.T

    # Scale features to have zero mean and unit variance
    scaler = StandardScaler()
    logmelspec_scaled = scaler.fit_transform(logmelspec_transposed)

    # Initialize a zero array with the target shape (L, D)
    logmel_padded = np.zeros((length, n_mels))

    # If the log-mel spectrogram is shorter than L frames, pad it with zeros
    if logmelspec_transposed.shape[0] < length:
        logmel_padded[:logmelspec_transposed.shape[0], :] = logmelspec_transposed
    # If it's longer, we'll truncate it to fit
    else:
        logmel_padded = logmelspec_transposed[:length, :]

    # Add an additional dimension to mimic a single-channel image for Conv2D
    logmel_padded = logmel_padded[..., np.newaxis]

    print(logmel_padded.shape)

    return logmel_padded
