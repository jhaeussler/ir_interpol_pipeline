import os
import matplotlib.pyplot as plt
import librosa
import pandas as pd
import numpy as np
from src.delux_logger import DeluxLogger, LogLevel
from multiprocessing import Pool


RUN_DEBUG_CODE = False


LOG = DeluxLogger()
LOG.set_level(LogLevel.DEBUG)


# 256 / 48000 Samples/s = 5.3 ms time window ... i dont care much for frequency res in this particular case.
# IRs are time-critical. The spectra are therefore never used in the ml-pipeline, I just added them for fun.
SFFT_WINDOW_SIZE = 256


def normalize_audio_in_dataset(dataset):
    """
    This function calls the normalize_signals func for all IR-Recording entries in the provided dataset.
    """
    LOG.info('Normalizing to highest amplitude in Data Set...')
    # makes the range of values (wertebereich) of ENTIRE dataset available to training process
    maximum = max(dataset.data['max_ir_vol_value'])

    datapoints_and_max_value = [(datapoint, maximum) for datapoint in dataset.data['irs']]

    with Pool(os.cpu_count()) as multiproc_pool:
        dataset.data['irs_normalized_to_entire_dataset'] = pd.Series(
            list(multiproc_pool.map(
                normalize_signals_to_highest_value_in_dataset, datapoints_and_max_value)
            )
        )


def normalize_signals_to_highest_value_in_dataset(loudspeaker_recs_and_maximum):
    loudspeaker_signals, maximum = loudspeaker_recs_and_maximum
    loudspeaker_signals = np.array(loudspeaker_signals)

    return loudspeaker_signals / maximum


def normalize_all_signals_to_one(signals):
    """
    Normalize all signals contained in the input. Functions as a wrapper for the normalize_audio_data funktion.
    """
    return np.array([normalize_audio_signal_to_one(signal) for signal in signals])


def normalize_audio_signal_to_one(signal):
    """
    Return the normalized signal. Normalization done with librosa.
    """
    normalized_signal = librosa.util.normalize(signal)
    if RUN_DEBUG_CODE:
        LOG.debug(f'Max value in Signal befor normalization: {max(abs(signal))}')
        LOG.debug(f'Max value in Signal after normalization: {max(abs(normalized_signal))}')
    return normalized_signal


def draw_plot(signal, sr=48000, title='PLot', save=False, path='', filename='plot', limit=0.7):
    """
    This Draws the plot of a Signal to the screen.
    """
    fig, ax = plt.subplots()
    length = np.arange(len(signal)) / sr

    ax.set_ylim(-limit, limit)
    ax.plot(length, signal)
    ax.set(title=title)
    ax.label_outer()
    ax.set_xlabel('Zeit in s')
    ax.set_ylabel('X(t)')

    fig.set_size_inches(6, 4)

    if save:
        if path != '':
            path += '/'
        fig.savefig(path + filename + '.png', dpi=150)
    else:
        plt.show()

    plt.close()


def draw_multiple_plots(signals, plotnames=None, sr=48000, save=False, path='', filename='multiplot', limit=0.7):
    if plotnames is None:
        plotnames = ['plot' for _ in signals]
    else:
        if len(plotnames) < len(signals):
            add_plotnames = len(signals) - len(plotnames)
            while add_plotnames > 0:
                plotnames.append('plot')
                add_plotnames -= 1

    fig, axs_array = plt.subplots(nrows=signals.shape[0], ncols=1, sharex='all')
    length = np.arange(len(signals[0])) / sr

    for index, row in enumerate(axs_array):
        row.set_ylim(-limit, limit)
        row.plot(length, signals[index])
        row.set(title=f'{plotnames[index]}')
        row.set_ylabel('X(t)')
    axs_array[-1].set_xlabel('Zeit in s')

    fig.set_size_inches(10, 6)
    plt.subplots_adjust(hspace=0.4)

    if save:
        fig.savefig(f'{path}/{filename}.png', dpi=150)
    else:
        plt.show()

    plt.close()


def detectect_signal_start_wrapper(signal):
    """
    this just controls the actual detect_signal_start func which is recursive and therefor not suited for debug
    printing.
    """
    index, _ = detect_signal_start(signal)
    # new ir signal first 2000
    if RUN_DEBUG_CODE:
        LOG.debug(f'Split Index: {index}')
        draw_plot(signal[index:index + 1500])

    return index


# 1e+6 triggers at Energy factor of 1000000 for the next window in respect to the previous window.
# Its devided by some value for every recursion when no difference that high was found
# between prev_samples and next_samples, see below for detail
def detect_signal_start(signal, window_size=150, threshold=1e+6, step=10):
    """
    This detects the first sudden increase in signal energy (the arrival time of the direct sound signal) by evaluating
    mean values of samples in a defined window (window size * 2).

    :param signal: The input signal

    :param window_size: The size of one half of the window. The window is split in two equal parts: past_samples and
    next_samples. The energy within these windows is compared. If the window is too small, a few samples of the
    IR (depending on samlping frequency) will be cut into the silent part as well. Default: 400

    :param threshold: The ratio by which the energy in the right half of the window needs to be bigger then the left
    side. This function calls itself recursively when the current ratio is not reached in the first quarter of the
    signal. The threshold is then decreased. You should probably just leave it at the high default value.

    :param step: amount of samples the window is shifted by

    return: The index at which the sigal should be split between Silence and IR.
    """
    index_of_signal_start = 0
    # will be set to true when threshold is reached in any recursion
    found = False

    # THis function is kind of dangerous because it could very well loop forever (while true and recursion).
    while True:

        # First check that enough samples lie before index_of_signal_start to create the past_samples window:
        # If index_of_signal_start < window, there are not enough previous samples yet, so pad with the first entry in
        # signal (0.0 wont work, because then mean_next / mean_prev may be high between padded 0.0 and Noise-floor)
        if index_of_signal_start < window_size:
            # pad with first value of signal so that there are enough samples before index_of_signal_start
            # past_samples = np.array([signal[0] for _ in range(window_size - index_of_signal_start)])
            past_samples = signal[:window_size - index_of_signal_start]
            # if index_of_signal_start 0, window is completely padded. Else add Samples up to index_of_signal_start to
            # the padded array.
            if index_of_signal_start > 0:
                past_samples = np.concatenate([past_samples, signal[:index_of_signal_start]])

        # No padding necessary, just copy samples from [index_of_signal_start - window_size] to (not incl.ding)
        # [index_of_signal_start] as past_samples
        else:
            past_samples = signal[index_of_signal_start - window_size:index_of_signal_start]

        # get the window of next_samples
        next_samples = signal[index_of_signal_start:index_of_signal_start + window_size]

        # Calculate the mean values of the previous samples' energies and the next samples energies
        mean_next = np.mean(abs(next_samples ** 2))
        mean_prev = np.mean(abs(past_samples ** 2))

        # get the ratio between the mean_vals of samples in previous and next window.
        if ((mean_next / mean_prev) > threshold) and mean_next > mean_prev:
            if RUN_DEBUG_CODE:
                print(f'mean_diff: {mean_next / mean_prev}, next: {mean_next}, prev: {mean_prev}')
            return index_of_signal_start, True

        # Only search the first third of signal.
        if index_of_signal_start >= (len(signal) / 2):
            if threshold > 1000:
                # ...start again with threshold / 4
                index_of_signal_start, found = detect_signal_start(signal, threshold=threshold * 0.25)
            else:
                # ...start again with threshold / 2
                index_of_signal_start, found = detect_signal_start(signal, threshold=threshold * 0.5)

        if found:
            return index_of_signal_start, found
        index_of_signal_start += step
