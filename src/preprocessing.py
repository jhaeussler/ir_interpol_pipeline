import matplotlib.pyplot as plt
import librosa
import numpy as np
from src.delux_logger import DeluxLogger, LogLevel


RUN_DEBUG_CODE = False


LOG = DeluxLogger()
LOG.set_level(LogLevel.DEBUG)


# 256 / 48000 Samples/s = 5.3 ms time window ... i dont care much for frequency res in this particular case.
# IRs are time-critical. The spectra are therefore never used in the ml-pipeline, I just added them for fun.
SFFT_WINDOW_SIZE = 256


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
