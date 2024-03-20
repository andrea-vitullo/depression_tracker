import numpy as np
import scipy.signal as sp
import scipy.fft as fft
from scipy.signal import argrelmax
import librosa
import os

import my_config
from utils import file_utils


def vad_cgd(wave, fs, duration_limit=100):
    """
    Extracts chirp group delay features from the speech signal.

    Args:
        wave (np.array): Waveform of the audio signal
        fs (int): Sampling rate of the audio signal.
        duration_limit (int): The limit for duration in seconds to process an audio file. Default is 100 seconds.

    Returns:
        np.array: chirp group delay spectrum of each frame
        np.array: peak indices of each frame
    """

    wave_length_seconds = len(wave) / fs
    if wave_length_seconds > duration_limit:
        wave = wave[:int(duration_limit * fs)]

    framel = 30
    frameshift = 10
    nfft = 1024

    start = 1
    stop = round(framel / 1000 * fs)
    shift = round(frameshift / 1000 * fs)
    cgd_results = []
    peak_indices_results = []  # New list to store peak indices

    wave = sp.lfilter([1, -0.97], 1, wave)

    while stop <= min(len(wave), int(duration_limit * fs)):
        seg = np.pad(wave[start:stop], (0, nfft - len(wave[start:stop])))
        win = np.blackman(len(seg))
        seg *= win

        cgd = chirp_group_delay(seg, fs)
        peak_indices = argrelmax(cgd)  # Calculate peak indices separately

        cgd_results.append(cgd)
        peak_indices_results.append(peak_indices)  # Store peak indices in its own list

        start += shift
        stop += shift

    chirp_results = np.array(cgd_results)

    chirp_results -= np.mean(chirp_results, axis=0)

    return cgd_results, peak_indices_results


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

    return chirp_group_delay  # Return only the group delays


def post_process(formant_peaks, fs=16000, max_formant_delta=250, min_formant_diff=50):
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

        # Discard formants that are too close together
        sorted_peaks = np.sort(current_peaks)
        too_close = np.where(np.diff(sorted_peaks) < min_formant_diff)[0]
        current_peaks[too_close] = 0

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


for path, directories, files in os.walk(my_config.DIRECTORY):
    for audio in files:
        if audio.endswith(my_config.FINAL_FORMAT):
            full_audio_path = os.path.join(path, audio)
            print(f"Extracting chirp group delay features from: {full_audio_path}...")
            wave, fs = librosa.load(full_audio_path, sr=16000)
            group_delays, peak_indices = vad_cgd(wave, int(fs))  # Unpack the returned tuple

            processed_formants_results = []  # To store processed peaks for each frame
            for frame_peak_indices in peak_indices:
                # Convert peak indices to frequency values
                frame_peak_indices_array = np.array(frame_peak_indices[0])  # Extract numpy array from the tuple
                frame_peak_freqs = (frame_peak_indices_array / len(wave)) * fs
                # Convert the list to a 2D numpy array before processing
                frame_peak_freqs_2D = np.array([frame_peak_freqs])
                processed_formants = post_process(frame_peak_freqs_2D, int(fs))

                # Flatten the 2D numpy array into 1D
                processed_formants_flattened = processed_formants.flatten()
                processed_formants_results.append(processed_formants_flattened)

            file_utils.save_outputs(processed_formants_results, audio, path=my_config.FORMANT_FEATURES)
