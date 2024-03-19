import numpy as np
import scipy.signal as sp
import scipy.fft as fft
from scipy.signal import argrelmax
import librosa
import pandas as pd
import os

import my_config


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
    cgd_results = []

    wave = sp.lfilter([1, -0.97], 1, wave)

    while stop <= len(wave):
        seg = wave[start:stop]
        seg = np.pad(seg, (0, nfft - len(seg)))
        win = np.blackman(len(seg))
        seg *= win

        cgd = chirp_group_delay(seg, fs)  # Modified
        cgd_results.append(cgd)  # Modified

        start += shift
        stop += shift

    chirp_results = np.array(cgd_results)

    chirp_results -= np.mean(chirp_results, axis=0)

    return chirp_results


def chirp_group_delay(frame, fs):
    nfft = 1024
    win = np.blackman(len(frame))
    frame *= win
    frame = np.pad(frame, (0, nfft - len(frame)))  # zero-padding
    R = 1.12 # Scaling factor (you may need to adjust this)

    n = np.arange(nfft)
    exponential_envelope = np.exp(-np.log(R) * n)
    frame *= exponential_envelope

    # Chirp Z-transform computation using FFT
    fourier_trans = fft.fft(frame, n=fs)
    ang_fft = np.angle(fourier_trans)

    # Group delay is minus derivative of the phase
    chirp_group_delay = -np.diff(ang_fft)

    # Local maxima
    peak_indices = argrelmax(chirp_group_delay)

    return chirp_group_delay, peak_indices


def post_process(formant_peaks, fs=16000, max_formant_delta=250):
    # Number of frames and formants
    num_frames, num_formants = formant_peaks.shape

    # Cost matrix initialization
    formant_peaks_cost = np.zeros(formant_peaks.shape)

    # Compute costs
    for kk in range(2, num_frames - 2):
        pre_pre_peaks = formant_peaks[kk - 2, :]
        post_post_peaks = formant_peaks[kk + 2, :]
        pre_peaks = formant_peaks[kk - 1, :]
        post_peaks = formant_peaks[kk + 1, :]
        current_peaks = formant_peaks[kk, :]

        current_peaks_cost = np.zeros(current_peaks.shape)

        for mm in range(num_formants):
            if (current_peaks[mm] == 0):
                current_peaks_cost[mm] = fs / 2
            else:
                # Search for closest matches
                distance_array_pre = np.sort(np.abs(pre_peaks - current_peaks[mm]))
                distance_array_post = np.sort(np.abs(post_peaks - current_peaks[mm]))
                distance_array_pre_pre = np.sort(np.abs(pre_pre_peaks - current_peaks[mm]))
                distance_array_post_post = np.sort(np.abs(post_post_peaks - current_peaks[mm]))

                all_distances = np.minimum((distance_array_pre[0] + distance_array_post[0]) / 2,
                                           [distance_array_pre[0], distance_array_post[0]])
                all_distances2 = np.minimum((distance_array_pre_pre[0] + distance_array_post_post[0]) / 2,
                                            [distance_array_pre_pre[0], distance_array_post_post[0]])

                current_peaks_cost[mm] = np.min(np.concatenate((all_distances, all_distances2)))

        formant_peaks_cost[kk, :] = current_peaks_cost

    # Apply filtering based on costs
    for kk in range(num_frames):
        for mm in range(num_formants):
            if formant_peaks_cost[kk, mm] > max_formant_delta:
                formant_peaks[kk, mm] = 0

    # Replace possible continuation values instead of zero values
    for kk in range(1, num_frames - 1):
        current_peaks = formant_peaks[kk, :]
        non_zero_indices = np.nonzero(current_peaks)[0]  # Finds non-zero elements
        non_zero_formants = current_peaks[non_zero_indices]
        num_non_zero_formants = len(non_zero_indices)
        num_zero_formants = num_formants - num_non_zero_formants

        if num_non_zero_formants < num_formants and num_non_zero_formants > 0:
            possible_values = np.sort(np.concatenate((formant_peaks[kk - 1, :], formant_peaks[kk + 1, :])))
            possible_values = possible_values[possible_values > 0]  # Discard zero entries
            possible_candidates = []

            for mm in range(len(possible_values)):
                distance_array = np.sort(np.abs(non_zero_formants - possible_values[mm]))
                if len(distance_array) == 0 or distance_array[
                    0] > max_formant_delta:  # This possible value not found in the current vector
                    possible_candidates.append(possible_values[mm])

            # Choose among possible candidates
            len_possible_candidates = len(possible_candidates)

            if len_possible_candidates <= num_zero_formants:
                current_peaks = sorted(non_zero_formants + possible_candidates + [0] * (
                            num_zero_formants - len_possible_candidates))
            elif num_zero_formants == 1:  # The most common case
                idx = np.argmin(np.diff(possible_candidates))  # index of smallest difference
                current_peaks = sorted(non_zero_formants + [possible_candidates[idx]])
            elif num_zero_formants < len_possible_candidates:
                possible_candidates = possible_candidates[:num_zero_formants]
                current_peaks = sorted(non_zero_formants + possible_candidates)

            formant_peaks[kk, :] = current_peaks  # Update this frame's formants

    return formant_peaks


def save_outputs(features, audio, path):
    """Save features to a CSV file."""

    df = pd.DataFrame(features)
    output_files = os.path.join(path, f'{audio[0:3]}_features.csv')
    df.to_csv(output_files, index=False)
    print(f"Saving .csv files: {output_files}")