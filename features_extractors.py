import numpy as np
import librosa

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


def extract_mfcc_features(audio, sr, d=N_MFCC, length=MAX_LENGTH_MFCC):
    # Compute MFCC features with the desired number of coefficients (D)
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=d, hop_length=MFCC_HOP_LENGTH, n_fft=N_FTT)

    # Transpose MFCCs to shape (time_steps, D) to align with Conv2D input expectations
    mfccs_transposed = mfccs.T

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


