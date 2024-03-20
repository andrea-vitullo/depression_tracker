import numpy as np
import scipy.signal as sp
import librosa
import os

import my_config
from utils import file_utils


def vad_mfcc(wave, fs):
    """
    Extracts MFCC features from the speech signal

    Args:
        wave (np.array): Waveform of the audio signal
        fs (int): Sampling rate of the audio signal.

    Returns:
        np.array: MFCC features
    """

    framel = 30
    frameshift = 10
    nfft = 1024

    start = 1
    stop = round(framel / 1000 * fs)
    shift = round(frameshift / 1000 * fs)
    mfcc_results = []

    wave = sp.lfilter([1, -0.97], 1, wave)

    while stop <= len(wave):
        seg = np.pad(wave[start:stop], (0, nfft - len(wave[start:stop])))
        if len(seg) < nfft:
            seg = np.pad(seg, (0, nfft - len(seg)))
        win = np.blackman(len(seg))
        seg *= win

        mfcc = librosa.feature.mfcc(y=seg, sr=fs, n_mfcc=my_config.N_MFCC, n_fft=nfft).mean(axis=1)
        mfcc_results.append(mfcc)

        start += shift
        stop += shift

    mfcc = np.array(mfcc_results)

    mfcc -= np.mean(mfcc, axis=0)

    return mfcc.T


for path, directories, files in os.walk(my_config.DIRECTORY):
    for audio in files:
        if audio.endswith(my_config.FINAL_FORMAT):
            full_audio_path = os.path.join(path, audio)
            print(f"Extracting MFCCs from: {full_audio_path}...")
            waveform, fs = librosa.load(full_audio_path, sr=16000)
            features = vad_mfcc(waveform, fs)

            file_utils.save_outputs(features, audio, path=my_config.MFCC_FEATURES)
