import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import librosa
from librosa.display import specshow
from multiprocessing import Pool
import os

global LOG


def initialize_log(log):
    global LOG
    LOG = log


AUDIO_INPUT_LEN = 4096
AUDIO_OUTPUT_LEN = 2048


def get_audio_sequence_dataset(dataset, ls_pos=0):
    all_sequences = []
    sequences_per_datapoint = []

    for _, datapoint in dataset.iterrows():
        signals = datapoint[0][ls_pos]

        with Pool(os.cpu_count()) as multiproc_pool:
            sequences_datapoint = list(multiproc_pool.map(sequence_signal, signals))

        all_sequences.append(np.concatenate(sequences_datapoint, axis=0))
        sequences_per_datapoint.append(np.array(sequences_datapoint))

    all_sequences = np.concatenate(all_sequences, axis=0)
    print(all_sequences.shape)
    print(sequences_per_datapoint[0].shape)

    return all_sequences, sequences_per_datapoint


def sequence_signal(signal):
    signal = signal[3500:63500]
    index = 0
    sequences = []
    while index < len(signal) - AUDIO_INPUT_LEN - AUDIO_OUTPUT_LEN:
        sequences.append(signal[index:index + AUDIO_INPUT_LEN + AUDIO_OUTPUT_LEN])
        index += AUDIO_OUTPUT_LEN

    return np.array(sequences)


def make_audio_dataset(sequences):
    input_label_pairs = list(map(split_audio_input_label_pairs, sequences))
    input_label_pairs = np.array(input_label_pairs)

    inputs = input_label_pairs[:, 0]
    labels = input_label_pairs[:, 1]

    inputs = np.concatenate(inputs, axis=0)
    inputs = np.reshape(inputs, [-1, AUDIO_INPUT_LEN, 1])
    labels = np.concatenate(labels, axis=0)
    labels = np.reshape(labels, [-1, AUDIO_OUTPUT_LEN, 1])
    print(f'shapes: {inputs.shape}, {labels.shape}')

    return tf.data.Dataset.from_tensor_slices((inputs, labels))


def split_audio_input_label_pairs(sequence):
    return sequence[:AUDIO_INPUT_LEN], sequence[AUDIO_INPUT_LEN:]


def make_datasets_from_pd_dataframe(dataset, column_name, split=0.9):
    column_of_interest = dataset.data[[column_name]]
    middle_of_data = 6 # int(len(column_of_interest) / 2)

    test_len = int(len(dataset) * 0.1)
    val_len = int(len(column_of_interest) * 0.1)

    train = column_of_interest.loc[:middle_of_data - 1]

    test = column_of_interest.loc[middle_of_data:middle_of_data + test_len - 1]
    val = column_of_interest.loc[middle_of_data + test_len:middle_of_data + test_len + val_len - 1]

    train = pd.concat(
        (train, column_of_interest.loc[(middle_of_data + test_len + val_len):int(len(column_of_interest) - 2)]), axis=0)

    print(f'length dataset: {len(dataset)}')
    print(f'length train: {len(train)}')
    print(f'length of val: {len(val)}')
    print(f'length of test: {len(test)}\n')

    return train, val, test


# CQT REVERB EXPERIMENT

# Max hop_length 2048
CQT_HOP_LENGTH = 512
N_BINS_PER_OCT = 128  # int(CQT_HOP_LENGTH / 2)
N_BINS = N_BINS_PER_OCT * 9  # 16000+ Hertz -> 9 Octaves above 32 Hz


def set_cqt_hop_length(new_hop_length):
    global CQT_HOP_LENGTH, N_BINS, N_BINS_PER_OCT

    CQT_HOP_LENGTH = new_hop_length
    N_BINS_PER_OCT = 128
    N_BINS = N_BINS_PER_OCT * 9  # 16000+ Hertz -> 9 Octaves above 32 Hz


def get_cqts(dataset, windwow_length, ls_pos=0):
    cqts = None
    cqts_datapoints = []

    set_cqt_hop_length(windwow_length)

    for _, datapoint in dataset.iterrows():

        signals = datapoint[0][ls_pos]
        with Pool(os.cpu_count()) as multiproc_pool:
            cqts_datapoint = list(multiproc_pool.map(cqt_of_entire_signal, signals))

        out_len = cqts_datapoint[0].shape[1]
        cqts_datapoint = np.concatenate(cqts_datapoint, axis=-1)
        if cqts is None:
            cqts = cqts_datapoint
        else:
            cqts = np.concatenate((cqts, cqts_datapoint), axis=-1)
        # draw_cqt(cqts_datapoint[:, :int(cqts_datapoint.shape[-1] / 7)])
        cqts_datapoints.append(np.reshape(np.transpose(cqts_datapoint), (7, -1, cqts_datapoint.shape[0])))

    return np.transpose(cqts), np.array(cqts_datapoints), out_len


def cqt_of_entire_signal(signal, signal_start=2048):
    interesting_signal = signal[signal_start:signal_start + 60000]
    return librosa.cqt(interesting_signal,
                       sr=48000,
                       hop_length=CQT_HOP_LENGTH,
                       n_bins=N_BINS,
                       bins_per_octave=N_BINS_PER_OCT)


def signal_from_cqts(cqt_signal, hop_length, bins_per_oct=128):
    return librosa.icqt(cqt_signal,
                        sr=48000,
                        hop_length=hop_length,
                        bins_per_octave=bins_per_oct)


def make_complex(sequence):
    return tf.complex(sequence[:, :, 0], sequence[:, :, 1])


def split_complex_tensor_to_two_dims(complex_input_array):
    real = tf.expand_dims(np.real(complex_input_array), 1)
    img = tf.expand_dims(np.imag(complex_input_array), 1)
    return tf.concat([real, img], axis=1)


def draw_cqt(cqt, hop_length, sr=48000, save=False, path='', filename='cqt_plot', bins_per_oct=128):
    abs_cqt = librosa.amplitude_to_db(np.abs(cqt))
    fig, ax = plt.subplots()

    img = librosa.display.specshow(abs_cqt,
                                   sr=sr,
                                   x_axis='time',
                                   y_axis='cqt_hz',
                                   ax=ax,
                                   hop_length=hop_length,
                                   bins_per_octave=bins_per_oct)
    ax.set_title(filename)
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    fig.set_size_inches(6, 4)

    if save:
        fig.savefig(f'{path}/{filename}.png', dpi=150)
    else:
        plt.show()
    plt.close()


def draw_multiple_cqts(cqts, hop_length, plotnames=None, sr=48000, save=False, path='', filename='multiplot', bins_per_octave=128):
    if plotnames is None:
        plotnames = ['plot' for _ in cqts]
    else:
        if len(plotnames) < len(cqts):
            add_plotnames = len(cqts) - len(plotnames)
            while add_plotnames > 0:
                plotnames.append('plot')
                add_plotnames -= 1

    fig, axs_array = plt.subplots(nrows=cqts.shape[0], ncols=1, sharex='all')

    img = None
    for index, row in enumerate(cqts):
        img = librosa.display.specshow(librosa.amplitude_to_db(abs(row)),
                                       sr=sr,
                                       y_axis='cqt_hz',
                                       x_axis='time',
                                       ax=axs_array[index],
                                       hop_length=hop_length,
                                       bins_per_octave=bins_per_octave)
        axs_array[index].set(title=f'{plotnames[index]}')

    # fig.colorbar(img, ax=axs_array, format="%+2.0f dB")

    axs_array[-1].set_xlabel('Zeit in s')
    fig.set_size_inches(10, 6)
    plt.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.8, hspace=0.4)

    cb_ax = fig.add_axes([0.85, 0.1, 0.02, 0.8])
    cbar = fig.colorbar(img, cax=cb_ax, format="%+2.0f dB")

    if save:
        fig.savefig(f'{path}/{filename}.png', dpi=150)
    else:
        plt.show()

    plt.close()


def split_complex_cqts_to_two_dims(complex_input_array):
    windows = []
    for window in complex_input_array:
        real = tf.expand_dims(np.real(window), 1)
        img = tf.expand_dims(np.imag(window), 1)
        windows.append(tf.expand_dims(tf.concat([real, img], axis=1), axis=0))
    return tf.concat(windows, axis=0)


def get_max_of_inputs(inputs):
    max_values = []
    for entry in inputs:
        max_values.append(max(entry.flatten()))

    return max(max_values)


INPUT_LENGTH = 1
OUTPUT_LENGTH = 1


def set_input_and_output_length(input_len, output_len):
    global INPUT_LENGTH, OUTPUT_LENGTH
    INPUT_LENGTH = input_len
    OUTPUT_LENGTH = output_len


def split_inputs_from_labels(window):
    inputs = window[:INPUT_LENGTH]
    inputs = split_complex_cqts_to_two_dims(inputs)

    labels = window[INPUT_LENGTH:INPUT_LENGTH + OUTPUT_LENGTH]
    labels = split_complex_cqts_to_two_dims(labels)

    return [inputs, labels]


def split_dataset_to_windows(dataset, window_length, step):
    windows = []
    index = 0
    while index + window_length < len(dataset):
        windows.append(dataset[index:int(index + window_length)])
        index += step

    return windows


def make_cqt_dataset(dataset, input_length=4, output_length=2, max_val=None):
    set_input_and_output_length(input_len=input_length, output_len=output_length)

    windows = split_dataset_to_windows(dataset, window_length=int(input_length + output_length), step=output_length)

    input_label_pairs = list(map(split_inputs_from_labels, windows))
    input_label_pairs = np.array(input_label_pairs)

    # print(input_label_pairs.shape)

    inputs = input_label_pairs[:, 0]
    inputs = np.concatenate(inputs, axis=0)
    inputs = np.reshape(inputs, (-1, input_length, inputs.shape[-2] * inputs.shape[-1]))

    labels = input_label_pairs[:, 1]
    labels = np.concatenate(labels, axis=0)
    labels = np.reshape(labels, (-1, output_length, labels.shape[-2] * labels.shape[-1]))

    if max_val is None:
        max_val = get_max_of_inputs((inputs, labels))

    inputs /= max_val
    labels /= max_val

    # print(f'input: {inputs.shape}')
    # print(f'labels: {labels.shape}')

    return tf.data.Dataset.from_tensor_slices((inputs, labels)), max_val
# Early Signal Experiment


def make_datasets_with_pos_and_audio(dataset, column_name='irs_normalized_to_one', split=0.85):
    column_of_interest = dataset.data[[column_name, 'MicPos']]
    middle_of_data = int(len(column_of_interest) / 2)

    test_len = int(len(dataset) * 0.1)
    val_len = int(len(column_of_interest) * 0.1)

    train = column_of_interest.loc[:middle_of_data - 1]

    test = column_of_interest.loc[middle_of_data:middle_of_data + test_len - 1]
    val = column_of_interest.loc[middle_of_data + test_len:middle_of_data + test_len + val_len - 1]

    train = pd.concat(
        (train, column_of_interest.loc[(middle_of_data + test_len + val_len):int(len(column_of_interest) - 2)]), axis=0)

    print(f'length dataset: {len(dataset)}')
    print(f'length train: {len(train)}')
    print(f'length of val: {len(val)}')
    print(f'length of test: {len(test)}')

    return train, val, test


def get_input_label_pairs_for_embedding_audio(dataset, col_name, ls_pos=0, upsampling=4, end_index=2500):
    inputs = []
    labels = []

    startindex = 3500
    endindex = startindex + end_index
    for row_index, row in dataset.iterrows():
        # print(f'row_index: {row_index}')
        if type(row[col_name]) is float:
            continue

        for signal in row[col_name][ls_pos]:
            inputs.append([row_index * upsampling])
            labels.append(signal[startindex:endindex])

    return tf.data.Dataset.from_tensor_slices((inputs, labels))


def get_input_label_pairs_for_embedding(dataset, cqt_hop_len, col_name, ls_pos=0, upsampling=4, max_val=None):
    inputs = []
    labels = []

    for row_index, row in dataset.iterrows():
        # print(f'row_index: {row_index}')
        if type(row[col_name]) is float:
            continue

        """with multiprocessing.Pool(os.cpu_count()) as multiproc_pool:
            label_cqts = list(multiproc_pool.map(get_cqts_of_signal_starts, row[col_name][ls_pos]))"""

        label_cqts = list(map(get_cqts_of_signal_starts, row[col_name][ls_pos]))

        last_label = row[col_name][ls_pos][0]

        for label_cqt in label_cqts:
            inputs.append([row_index * upsampling])
            labels.append(label_cqt)

    if max_val is None:
        max_val = get_max_of_inputs(np.array(labels))
    # print(max_val)
    labels /= max_val

    return tf.data.Dataset.from_tensor_slices((inputs, labels)), max_val


def get_cqts_of_signal_starts(signal, cqt_window_size=256, end_index=2500):
    # You could do cqt on only the first 8000 samples or so, but cqt is most precise on longer signals
    interessantes_signal = signal[3500:65500]

    bins_per_oct = 128
    n_bins = bins_per_oct * 9
    cqt = librosa.cqt(interessantes_signal,
                      sr=48000,
                      hop_length=cqt_window_size,
                      n_bins=n_bins,
                      bins_per_octave=bins_per_oct)

    """reconstruct = librosa.icqt(cqt,
                               sr=48000,
                               hop_length=cqt_window_size,
                               bins_per_octave=bins_per_oct)"""

    # draw_cqt(cqt, hop_length=cqt_window_size)
    # draw_multiple_plots(np.array([interessantes_signal[:len(reconstruct)], reconstruct]))

    start_windows = int(end_index / cqt_window_size)
    cqt_start = np.transpose(cqt)[:start_windows]
    tf_label_cqt = split_complex_cqts_to_two_dims(cqt_start)
    tf_label_cqt = tf.reshape(tf_label_cqt, [cqt_start.shape[0], 2 * cqt_start.shape[1]])

    # print(tf_input_cqt.shape)
    # print(cqt.shape)
    # print(cqt_start.shape)

    return tf_label_cqt


def plot_batch_sizes(ds):
    batch_sizes = [batch[0].shape[0] for batch in ds]
    plt.bar(range(len(batch_sizes)), batch_sizes)
    plt.xlabel('Batch number')
    plt.ylabel('Batch size')
    plt.show()


# Richtungsvektorexperiment

def make_datasets_with_pos_and_directions(dataset, column_name='sdm_speaker1'):
    # print(dataset.data.columns)

    column_of_interest = dataset.data[[column_name, 'MicPos']]
    middle_of_data = int(len(column_of_interest) / 2)

    test_len = int(len(dataset) * 0.1)
    val_len = int(len(column_of_interest) * 0.1)

    train = column_of_interest.loc[:middle_of_data - 1]

    test = column_of_interest.loc[middle_of_data:middle_of_data + test_len - 1]
    val = column_of_interest.loc[middle_of_data + test_len:middle_of_data + test_len + val_len - 1]

    train = pd.concat(
        (train, column_of_interest.loc[(middle_of_data + test_len + val_len):int(len(column_of_interest) - 2)]), axis=0)

    print(f'length dataset: {len(dataset)}')
    print(f'length train: {len(train)}')
    print(f'length of val: {len(val)}')
    print(f'length of test: {len(test)}')

    return train, val, test


def sequence_richtungsvektoren(dataset, out_len):
    # print(dataset)
    col_name = 'sdm_speaker1'
    inputs = []
    labels = []
    upsampling = 4
    label_length = out_len

    max_sdm_value = 0.

    for row_index, row in dataset.iterrows():
        sdm = np.array(row[col_name]['sdm'])
        sdm = sdm[:label_length]

        inputs.append([row_index * upsampling])
        labels.append(sdm)

        if max_sdm_value < max(abs(sdm.flatten())):
            max_sdm_value = max(abs(sdm.flatten()))

    labels = np.array(labels)
    labels /= max_sdm_value

    return tf.data.Dataset.from_tensor_slices((inputs, labels)), max_sdm_value


def sequence_richtungsvektoren_dims_separat(dataset):
    # print(dataset)
    col_name = 'sdm_speaker1'
    inputs = []
    labels_x = []
    labels_y = []
    labels_z = []
    upsampling = 4
    label_length = 2500

    max_sdm_value = 0.

    for row_index, row in dataset.iterrows():
        sdm = np.array(row[col_name]['sdm'])
        sdm = sdm[:label_length]

        if max_sdm_value < max(abs(sdm.flatten())):
            max_sdm_value = max(abs(sdm.flatten()))

        sdm_x = sdm[:, 0]
        sdm_y = sdm[:, 1]
        sdm_z = sdm[:, 2]

        inputs.append([row_index * upsampling])
        labels_x.append(sdm_x)
        labels_y.append(sdm_y)
        labels_z.append(sdm_z)

    labels_x = np.array(labels_x) / max_sdm_value
    labels_y = np.array(labels_y) / max_sdm_value
    labels_z = np.array(labels_z) / max_sdm_value

    return tf.data.Dataset.from_tensor_slices((inputs, labels_x)), \
           tf.data.Dataset.from_tensor_slices((inputs, labels_y)), \
           tf.data.Dataset.from_tensor_slices((inputs, labels_z)), max_sdm_value
