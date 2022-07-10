import datetime
import os

from scipy.io.wavfile import write
import numpy as np

from src.ml_architectures.reverb_generator_model import ReverbGenerator
from src.ml_architectures.ir_generator_model import IRGenerator
import src.prepare_for_neural_nets as prep
from src.preprocessing import draw_multiple_plots, draw_plot
import tensorflow as tf

global LOG

BATCH_SIZE = 32


def load_pretraind_models_and_predict_ir():

    early_ir_generator = tf.keras.Sequential()
    early_ir_generator = tf.keras.models.load_model('./trained_models/ir_generator.h5')

    reverb_generator = tf.keras.Sequential()
    reverb_generator = tf.keras.models.load_model('./trained_models/reverb_generator.h5')

    print(early_ir_generator.summary())
    print(reverb_generator.summary())

    input_pos = [[1]]
    inputs = [[index + 1] for index in range(40)]

    start_signals = []
    for next_input in inputs:
        input_pos = np.array([next_input])
        print(input_pos.shape)
        start_signal = early_ir_generator.predict(input_pos)
        start_signals.append(start_signal)

    for index, signal in enumerate(start_signals):

        rir_cqt = np.reshape(signal, [signal.shape[1], int(signal.shape[-1] / 2), 2])
        rir_cqt = np.transpose(prep.make_complex(rir_cqt))
        print(f'rirshape: {rir_cqt.shape}')

        out_cqt_length = rir_cqt.shape[1]

        input_signal = signal[:, 1:]
        label_length = 4
        norm = 2.1069295406341553

        predictions = []
        while out_cqt_length < 113:
            prediction = reverb_generator.predict(input_signal)

            input_signal = input_signal[:, int(label_length):]
            input_signal = tf.concat((input_signal, prediction), axis=1)

            prediction *= norm
            prediction_reshape = tf.reshape(prediction, [label_length, int(prediction.shape[-1] / 2), 2])
            prediction_cqt = np.transpose(prep.make_complex(prediction_reshape))
            rir_cqt = np.concatenate((rir_cqt, prediction_cqt), axis=-1)

            predictions.append(prediction_cqt)
            out_cqt_length = rir_cqt.shape[-1]

        prep.draw_cqt(rir_cqt, save=True, filename=f'{index + 1}_rir', hop_length=256, path='./trained_models/rirs/')
        rir = prep.signal_from_cqts(rir_cqt, hop_length=256)
        draw_plot(rir, save=True, filename=f'{index + 1}_rir_audio', limit=1.2, path='./trained_models/rirs/')

        write(f'./trained_models/rirs/{index + 1}_rir.wav', 48000, rir)


def richtungsvektorexperiment(dataset, model_struct=0, num_of_epochs=1, learning_rate=0.00001):
    # Sehr kleiner Datensatz..
    train, val, test = prep.make_datasets_with_pos_and_directions(dataset)

    dims_einzeln = True

    input_length = 1
    input_dim = 1
    output_length = 2500

    if dims_einzeln:
        output_dims = 1
    else:
        output_dims = 3

    upsampling_factor = 4
    datasetlen = len(dataset)

    if dims_einzeln:
        train_ds, train_ds_y, train_ds_z, _ = prep.sequence_richtungsvektoren_dims_separat(train)
        val_ds, val_ds_y, val_ds_z, _ = prep.sequence_richtungsvektoren_dims_separat(val)
        test_ds, test_ds_y, test_ds_z, _ = prep.sequence_richtungsvektoren_dims_separat(test)
    else:
        train_ds, _ = prep.sequence_richtungsvektoren(train, output_length)
        val_ds, _ = prep.sequence_richtungsvektoren(val, output_length)
        test_ds, _ = prep.sequence_richtungsvektoren(test, output_length)

    experiment_info = f'Length of train: {len(train)}\n' \
                      f'batch size: {BATCH_SIZE}\n' \
                      f'num of epochs: {num_of_epochs}\n' \
                      f'learning rate: {learning_rate}\n' \
                      f'train elem spec:\n{train_ds.element_spec}\n'

    LOG.info(experiment_info)

    batch_size_rv = 4

    ir_generator = IRGenerator(input_sequence_length=input_length,
                               feature_dims=input_dim,
                               batch_size=batch_size_rv,
                               learning_rate=learning_rate,
                               out_sequence_length=output_length,
                               output_feature_dims=output_dims,
                               model_struct=model_struct,
                               # one pos was dropped because of nans
                               max_rec_pos_index=int((datasetlen - 1) * upsampling_factor))

    start_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    experiment_path = './experiments/early_signal/' + start_time + '_experiment'
    os.mkdir(experiment_path)

    with open(f'{experiment_path}/properties.txt', 'w') as fp:
        ir_generator.model.summary(print_fn=lambda x: fp.write(x + '\n'))
        fp.write('\n')
        fp.write(f'input length: {input_length}\n')
        fp.write(f'label length: {output_length}\n')
        fp.write(f'training epochs: {num_of_epochs}')
        fp.write(f'start time: {start_time}')
        fp.write(experiment_info)

    """tf.keras.utils.plot_model(
        reverb_generator.model,
        to_file=f"{experiment_path}/model.png",
        show_shapes=True,
        show_dtype=False,
        show_layer_names=False,
        rankdir="TB",
        expand_nested=False,
        dpi=120,
        layer_range=None,
        show_layer_activations=False,
    )"""

    LOG.info('Training model...\n')

    train_ds_batched = train_ds.batch(batch_size_rv)
    val_ds_batched = val_ds.batch(batch_size_rv)

    ir_generator.train_model(train_ds_batched, val_ds_batched, epochs=num_of_epochs)

    # reverb_generator.model.save(f'{experiment_path}/model.h5')

    LOG.info('\n...Done. Generating Predictions...')

    iteri = test_ds.as_numpy_iterator()

    if dims_einzeln:
        for index, input_label_pair in enumerate(iteri):
            prediction = ir_generator.predict(input_label_pair[0])
            prediction = prediction.flatten()

            label = input_label_pair[1].flatten()
            diff = label - prediction

            normed_pred = np.clip(prediction, -1, 1)
            diff_norm = label - normed_pred

            draw_multiple_plots(
                np.concatenate((np.expand_dims(label, 0),
                                np.expand_dims(prediction, 0),
                                np.expand_dims(diff, 0)), axis=0),
                limit=1.25,
                plotnames=['label', 'prediction', 'difference'],
                filename=f'{index}_RV-Plot',
                path=experiment_path,
                save=True)

            draw_multiple_plots(
                np.concatenate((np.expand_dims(label, 0),
                                np.expand_dims(normed_pred, 0),
                                np.expand_dims(diff_norm, 0)), axis=0),
                limit=1.25,
                plotnames=['label', 'clipped_pred', 'difference'],
                filename=f'{index}_RV-Plot_clipped',
                path=experiment_path,
                save=True)

            draw_multiple_plots(
                np.concatenate((np.expand_dims(label[:1000], 0),
                                np.expand_dims(normed_pred[:1000], 0),
                                np.expand_dims(diff_norm[:1000], 0)), axis=0),
                limit=1.25,
                plotnames=['label', 'clipped_pred', 'difference'],
                filename=f'{index}_RV-Plot_clipped_small',
                path=experiment_path,
                save=True)

    else:
        for index, input_label_pair in enumerate(iteri):
            prediction = ir_generator.predict(input_label_pair[0])
            prediction = np.reshape(prediction, [prediction.shape[1], prediction.shape[2]])

            label = input_label_pair[1]
            # print(label.shape)
            diff = prediction - label

            # norm = max(prediction.flatten())
            # normed_pred = prediction / norm
            normed_pred = np.clip(prediction, -1, 1)
            print(max(normed_pred.flatten()))

            draw_multiple_plots(
                np.concatenate((np.expand_dims(normed_pred[:, 0], 0),
                                np.expand_dims(label[:, 0], 0),
                                np.expand_dims(diff[:, 0], 0)), axis=0),
                limit=1.5,
                plotnames=['pred_clip x', 'label x', 'diff x'],
                filename=f'{index}_prediction',
                path=experiment_path,
                save=True)
            draw_multiple_plots(
                np.concatenate((np.expand_dims(normed_pred[:, 1], 0),
                                np.expand_dims(label[:, 1], 0),
                                np.expand_dims(diff[:, 1], 0)), axis=0),
                limit=1.5,
                plotnames=['pred_clipped y', 'label y', 'diff y'],
                filename=f'{index}_label',
                path=experiment_path,
                save=True)

            draw_multiple_plots(
                np.concatenate((np.expand_dims(normed_pred[:, 2], 0),
                                np.expand_dims(label[:, 2], 0),
                                np.expand_dims(diff[:, 2], 0)), axis=0),
                limit=1.5,
                plotnames=['pred_clipped z', 'label z', 'diff z'],
                filename=f'{index}_diff',
                path=experiment_path,
                save=True)

            draw_multiple_plots(
                np.concatenate((np.expand_dims(prediction[:, 0], 0),
                                np.expand_dims(prediction[:, 1], 0),
                                np.expand_dims(prediction[:, 2], 0)), axis=0),
                limit=1.5,
                plotnames=['unclipped x', 'unclipped y', 'undclipped z'],
                filename=f'{index}_normed',
                path=experiment_path,
                save=True)


def generate_early_signal(dataset, model_struct=0, num_of_epochs=1, learning_rate=0.0001):
    """
    Dense: Wenige Schichten, mehr als 3 verschlechtert ergebnisse schnell
    """

    train, val, test = prep.make_datasets_with_pos_and_audio(dataset, column_name='irs_normalized_to_one')

    upsampling_factor = 4
    # train = train[::10]
    print(f'actaul train len: {len(train)}')

    datasetlen = len(dataset)

    print(learning_rate)

    cqt_transform = True
    # Also change hop len in prep.get_input_label_pairs_for_embedding->get_cqts_of_signal_starts
    cqt_hop_len = 256
    n_bins_per_oct = 128
    max_cqt_val = None
    if cqt_transform:
        train_split, max_cqt_val = prep.get_input_label_pairs_for_embedding(train, cqt_hop_len=cqt_hop_len,
                                                                            col_name='irs_normalized_to_one',
                                                                            upsampling=upsampling_factor)

        val_split, _ = prep.get_input_label_pairs_for_embedding(val, cqt_hop_len=cqt_hop_len,
                                                                col_name='irs_normalized_to_one',
                                                                upsampling=upsampling_factor,
                                                                max_val=max_cqt_val)

        test_split, _ = prep.get_input_label_pairs_for_embedding(test, cqt_hop_len=cqt_hop_len,
                                                                 col_name='irs_normalized_to_one',
                                                                 upsampling=upsampling_factor)

    else:
        train_split = prep.get_input_label_pairs_for_embedding_audio(train,
                                                                     col_name='irs_normalized_to_one',
                                                                     upsampling=upsampling_factor)

        val_split = prep.get_input_label_pairs_for_embedding_audio(val,
                                                                   col_name='irs_normalized_to_one',
                                                                   upsampling=upsampling_factor)

        test_split = prep.get_input_label_pairs_for_embedding_audio(test,
                                                                    col_name='irs_normalized_to_one',
                                                                    upsampling=upsampling_factor)

    first_entry = next(iter(train_split))
    input_length = first_entry[0].numpy().shape[0]
    input_dim = 1

    output_length = first_entry[1].shape[0]
    if cqt_transform:
        output_dims = first_entry[1].shape[1]
    else:
        output_dims = 1

    print(first_entry[1].shape)

    print(f'input shape: {first_entry[0].shape}')
    print(f'out_length: {output_length}')
    # print(f'out_dims: {output_dims}')

    train_batched = train_split.batch(BATCH_SIZE)
    val_batched = val_split.batch(BATCH_SIZE)

    ir_generator = IRGenerator(input_sequence_length=input_length,
                               feature_dims=input_dim,
                               batch_size=BATCH_SIZE,
                               learning_rate=learning_rate,
                               out_sequence_length=output_length,
                               output_feature_dims=output_dims,
                               model_struct=model_struct,
                               # one pos was dropped because of nans
                               max_rec_pos_index=int((datasetlen - 1) * upsampling_factor))

    LOG.info('Training model...\n')

    start_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    experiment_path = './experiments/early_signal/' + start_time + '_experiment'
    os.mkdir(experiment_path)

    experiment_info = f'cqt_windows_in_input: {input_length}\n' \
                      f'out_frequency_bins: {output_dims}\n' \
                      f'cqt_windows_in_output: {output_length}\n' \
                      f'batch size: {BATCH_SIZE}\n' \
                      f'num of epochs: {num_of_epochs}\n' \
                      f'learning rate: {learning_rate}\n' \
                      f'train elem spec:\n{train_batched.element_spec}\n'

    with open(f'{experiment_path}/properties.txt', 'w') as fp:
        ir_generator.model.summary(print_fn=lambda x: fp.write(x + '\n'))
        fp.write('\n')
        fp.write(f'input length: {input_length}\n')
        fp.write(f'label length: {output_length}\n')
        fp.write(f'label dims: {output_dims}\n')
        fp.write(f'training epochs: {num_of_epochs}')
        fp.write(f'start time: {start_time}')
        fp.write(experiment_info)

    """tf.keras.utils.plot_model(
        ir_generator.model,
        to_file=f"{experiment_path}/model.png",
        show_shapes=True,
        show_dtype=False,
        show_layer_names=False,
        rankdir="TB",
        expand_nested=False,
        dpi=120,
        layer_range=None,
        show_layer_activations=False,
    )"""

    ir_generator.train_model(train_batched, val_batched, epochs=num_of_epochs)
    # ir_generator.model.save(f'{experiment_path}/model.h5')
    LOG.info('...Done. Predicting Test Data...')

    iterator = iter(test_split)
    test_batch = [next(iterator) for _ in range(len(test_split))]

    predictions = []
    inputs = []
    labels = []
    for index, datapoint in enumerate(test_batch):
        if index % 7 == 0:
            # print(f'Datapoint: {datapoint}')
            inputs.append(datapoint[0])
            label = datapoint[1]

            if cqt_transform:
                if max_cqt_val is None:
                    print('error here.')
                label *= max_cqt_val

                label = prep.make_complex(np.reshape(label, [label.shape[-2], int(label.shape[-1] / 2), 2]))
                labels.append(np.transpose(label))
            else:
                labels.append(label)

            next_input = tf.expand_dims(datapoint[0], axis=0)
            next_prediction = ir_generator.predict(next_input)
            print(f'next_input_shape: {next_input.shape}')

            if cqt_transform:
                next_prediction *= max_cqt_val

            predictions.append(next_prediction)

    for index, prediction in enumerate(predictions):
        if cqt_transform:
            cqt = prediction[0]
            cqt = prep.make_complex(np.reshape(cqt, [cqt.shape[-2], int(cqt.shape[-1] / 2), 2]))
            cqt = np.transpose(cqt)

            cqt_diff = abs(abs(labels[index]) - abs(cqt))

            prep.draw_cqt(cqt,
                          hop_length=cqt_hop_len,
                          bins_per_oct=n_bins_per_oct,
                          filename=f'{index}_CQT_Prediction_Pos_{inputs[index]}',
                          path=experiment_path,
                          save=True)

            prep.draw_cqt(labels[index],
                          hop_length=cqt_hop_len,
                          bins_per_oct=n_bins_per_oct,
                          filename=f'{index}_CQT_Label_Pos_{inputs[index]}',
                          path=experiment_path,
                          save=True)

            prep.draw_cqt(cqt_diff,
                          hop_length=cqt_hop_len,
                          bins_per_oct=n_bins_per_oct,
                          filename=f'{index}_CQT_Difference_Pos_{inputs[index]}',
                          path=experiment_path,
                          save=True)

            prep.draw_multiple_cqts(np.concatenate(
                (np.expand_dims(labels[index], axis=0),
                 np.expand_dims(cqt, axis=0),
                 np.expand_dims(cqt_diff, axis=0)),
                axis=0),
                hop_length=cqt_hop_len,
                bins_per_octave=n_bins_per_oct,
                plotnames=['label', 'prediction', 'difference'],
                filename=f'{index}_CQT_Pos_{inputs[index]}',
                path=experiment_path,
                save=True)

            label_as_audio = prep.signal_from_cqts(labels[index], hop_length=cqt_hop_len, bins_per_oct=n_bins_per_oct)
            print(f'max:{max(label_as_audio.flatten())}')
            pred_as_audio = prep.signal_from_cqts(cqt, hop_length=cqt_hop_len, bins_per_oct=n_bins_per_oct)

            diff_as_audio = label_as_audio - pred_as_audio

            write(f'{experiment_path}/{index}_signal_predicted.wav', 48000, pred_as_audio)
            write(f'{experiment_path}/{index}_signal_label.wav', 48000, label_as_audio)

            draw_multiple_plots(np.concatenate(
                (np.expand_dims(label_as_audio, axis=0),
                 np.expand_dims(pred_as_audio, axis=0),
                 np.expand_dims(diff_as_audio, axis=0)),
                axis=0),
                save=True,
                path=experiment_path,
                filename=f'{index}_Audio_Pos_{inputs[index]}',
                limit=2.,
                plotnames=['label', 'prediction', 'difference'])

            pred_poses = [[4], [5], [6], [7], [8], [9]]

            for pos in pred_poses:
                new_input = np.array([pos])

                prediction = ir_generator.predict(new_input)
                prediction = prep.make_complex(
                    np.reshape(prediction, [prediction.shape[1], int(prediction.shape[2] / 2), 2]))

                prep.draw_cqt(np.transpose(prediction),
                              hop_length=cqt_hop_len,
                              bins_per_oct=n_bins_per_oct,
                              filename=f'embedded_prediction_of_real_Pos_{pos}',
                              path=experiment_path,
                              save=True)
        else:
            diff = labels[index].numpy().flatten() - prediction.flatten()
            draw_multiple_plots(
                np.concatenate([np.expand_dims(labels[index], 0), np.expand_dims(prediction.flatten(), 0),
                                np.expand_dims(diff, 0)], axis=0),
                filename=f'{index}_Prediction_Label_Pos_{inputs[index]}',
                path=experiment_path,
                save=True,
                limit=1.2,
                plotnames=['label', 'prediction', 'difference'])

            write(f'{experiment_path}/{index}_signal_predicted.wav', 48000, prediction.flatten())
            write(f'{experiment_path}/{index}_signal_label.wav', 48000, labels[index].numpy().flatten())

    LOG.info('...Done.')


def generate_reverb(dataset,
                    model_struct,
                    num_of_epochs,
                    sequence_length,
                    output_length,
                    learning_rate,
                    cqt_window_length=512):
    if model_struct is None:
        model_struct = 0

    LOG.info('splitting data...\n')
    train, val, test = prep.make_datasets_from_pd_dataframe(dataset, column_name='irs_normalized_to_one')
    LOG.info('...Done.')

    LOG.info('Sequencing and Transforming Data...')

    first_sequence = train.values[0][0][0][0]
    print(f'here: {first_sequence.shape}')
    first_sequence = first_sequence[3500:51500]

    draw_plot(first_sequence, save=True, filename='first_test_Signal', title='RIR aus He 2505', limit=1.2)

    run_audio = True
    if run_audio:
        run_audio_experiment(train, val, test,
                             sequence_length,
                             output_length,
                             model_struct,
                             num_of_epochs,
                             learning_rate)
    else:
        run_cqt_experiment(train, val, test,
                           sequence_length=sequence_length,
                           output_length=output_length,
                           model_struct=model_struct,
                           num_of_epochs=num_of_epochs,
                           cqt_window_length=cqt_window_length,
                           learning_rate=learning_rate)


def run_audio_experiment(train, val, test, sequence_length, output_length, model_struct, num_of_epochs, learning_rate):
    train = train[::20]

    train_sequences, _ = prep.get_audio_sequence_dataset(train)
    val_sequences, _ = prep.get_audio_sequence_dataset(val)
    test_sequences, test_sequences_per_datapoint = prep.get_audio_sequence_dataset(test)

    train_ds = prep.make_audio_dataset(train_sequences)
    val_ds = prep.make_audio_dataset(val_sequences)

    experiment_info = f'cqt_windows_in_input: {sequence_length}\n' \
                      f'frequency_bins: {1}\n' \
                      f'cqt_windows_in_output: {output_length}\n' \
                      f'Length of train: {len(train_ds)}\n' \
                      f'batch size: {BATCH_SIZE}\n' \
                      f'num of epochs: {num_of_epochs}\n' \
                      f'learning rate: {learning_rate}\n' \
                      f'train elem spec:\n{train_ds.element_spec}\n'

    LOG.info(experiment_info)

    reverb_generator = ReverbGenerator(input_sequence_length=sequence_length,
                                       feature_dims=1,
                                       learning_rate=learning_rate,
                                       out_sequence_length=output_length,
                                       model_struct=model_struct)

    start_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    experiment_path = './experiments/reverb/' + start_time + '_experiment'
    os.mkdir(experiment_path)

    with open(f'{experiment_path}/properties.txt', 'w') as fp:
        reverb_generator.model.summary(print_fn=lambda x: fp.write(x + '\n'))
        fp.write('\n')
        fp.write(f'input length: {sequence_length}\n')
        fp.write(f'label length: {output_length}\n')
        fp.write(f'training epochs: {num_of_epochs}')
        fp.write(f'start time: {start_time}')
        fp.write(experiment_info)

    """tf.keras.utils.plot_model(
        reverb_generator.model,
        to_file=f"{experiment_path}/model.png",
        show_shapes=True,
        show_dtype=False,
        show_layer_names=False,
        rankdir="TB",
        expand_nested=False,
        dpi=120,
        layer_range=None,
        show_layer_activations=False,
    )"""

    LOG.info('Training model...\n')

    train_ds_batched = train_ds.batch(BATCH_SIZE)
    val_ds_batched = val_ds.batch(BATCH_SIZE)

    reverb_generator.train_model(train_ds_batched, val_ds_batched, epochs=num_of_epochs)

    # reverb_generator.model.save(f'{experiment_path}/model.h5')

    LOG.info('\n...Done. Generating Batch Predictions...')

    # Reference Experiment

    predict_signal = test_sequences_per_datapoint[0][0]
    print(f'signalshape: {predict_signal.shape}')
    test_ds = prep.make_audio_dataset(predict_signal)
    print(test_ds.element_spec)

    test_ds = test_ds.batch(1)

    iteri = test_ds.as_numpy_iterator()

    first_input_label_pair = next(test_ds.as_numpy_iterator())
    first_input_sequence = first_input_label_pair[0]

    labels = [first_input_sequence.flatten()]
    batch_predictions = [first_input_sequence.flatten()]
    for input_label_pair in iteri:
        prediction = reverb_generator.predict(input_label_pair[0])
        batch_predictions.append(prediction.flatten())
        labels.append(input_label_pair[1].flatten())

    batch_predictions = np.concatenate(batch_predictions)
    labels = np.concatenate(labels)

    diff = labels - batch_predictions

    draw_plot(batch_predictions,
              filename=f'Prediction_Audio',
              title='prediction',
              path=experiment_path,
              save=True,
              limit=1.6, )
    draw_plot(labels,
              filename=f'Label_Audio',
              title='label',
              path=experiment_path,
              save=True,
              limit=1.6, )

    draw_plot(diff,
              filename=f'difference_Audio',
              title='difference',
              path=experiment_path,
              save=True,
              limit=1.6, )

    draw_multiple_plots(
        np.concatenate([np.expand_dims(labels, 0), np.expand_dims(batch_predictions, 0), np.expand_dims(diff, 0)],
                       axis=0),
        filename=f'Prediction_Label_Audio',
        path=experiment_path,
        save=True,
        limit=1.6,
        plotnames=['label', 'prediction', 'difference'])

    # Complete prediction Experiment

    output_signal = first_input_sequence.flatten()
    input_signal = first_input_sequence
    while len(output_signal) < len(labels):
        prediction = reverb_generator.predict(input_signal)
        output_signal = np.concatenate((output_signal, prediction.flatten()))

        input_signal = input_signal[:, prediction.shape[1]:]
        input_signal = np.concatenate((input_signal, prediction), axis=1)

    draw_plot(output_signal,
              filename=f'Complete_Prediction_Audio',
              path=experiment_path,
              save=True,
              limit=1.6, )

    diff_complete = labels - output_signal
    draw_plot(diff_complete,
              filename=f'Difference_Complete_Prediction_Audio',
              path=experiment_path,
              save=True,
              limit=1.6, )

    draw_multiple_plots(
        np.concatenate([np.expand_dims(labels, 0), np.expand_dims(output_signal, 0), np.expand_dims(diff_complete, 0)],
                       axis=0),
        filename=f'Complete_Prediction_Label_Audio',
        path=experiment_path,
        save=True,
        limit=1.6,
        plotnames=['label', 'prediction', 'difference'])

    write(f'{experiment_path}/signal_predicted_batch.wav', 48000, batch_predictions)
    write(f'{experiment_path}/signal_predicted_all.wav', 48000, output_signal)
    write(f'{experiment_path}/signal_label.wav', 48000, labels)


def run_cqt_experiment(train, val, test,
                       sequence_length=1,
                       output_length=1,
                       model_struct=0,
                       num_of_epochs=1,
                       cqt_window_length=512,
                       learning_rate=1e-4):
    # train = train[::20]

    LOG.info(f'Actual Train size: {len(train)}')

    """first_sequence = test.values[0][0][0]
    prep.draw_plot(first_sequence, save=True, filename='first_test_sequence', title='RIR aus He 1539b')"""

    train_cqts_0_all, _, cqt_out_len = prep.get_cqts(train, cqt_window_length, ls_pos=0)
    val_cqts_0_all, _, _ = prep.get_cqts(val, cqt_window_length)
    test_cqts_0_all, test_cqts_0_per_datapoint, _ = prep.get_cqts(test, cqt_window_length)

    LOG.info(f'Processed {len(train_cqts_0_all)} training samples.')
    LOG.info('Transform to Tensorflow shape...')

    input_ln = sequence_length
    output_ln = output_length

    train_ds_0_cqt, norm_train = prep.make_cqt_dataset(train_cqts_0_all,
                                                       input_length=input_ln,
                                                       output_length=output_ln)

    LOG.info('Training data loaded. Loading test and val...')
    val_ds_0_cqt, _ = prep.make_cqt_dataset(val_cqts_0_all, input_length=input_ln, output_length=output_ln,
                                            max_val=norm_train)
    test_ds_0_cqt, _ = prep.make_cqt_dataset(test_cqts_0_all, input_length=input_ln, output_length=output_ln,
                                             max_val=norm_train)


    print(f'norm_train: {norm_train}')
    LOG.info('...Done.')

    first_sequence = next(iter(train_ds_0_cqt))

    cqt_input_length = first_sequence[0].shape[0]
    feature_dims = first_sequence[0].shape[1]
    output_length = first_sequence[1].shape[0]

    experiment_info = f'cqt_window_size: {cqt_window_length}\n' \
                      f'cqt_windows_in_input: {cqt_input_length}\n' \
                      f'frequency_bins: {feature_dims}\n' \
                      f'cqt_windows_in_output: {output_length}\n' \
                      f'Length of train: {len(train_ds_0_cqt)}\n' \
                      f'batch size: {BATCH_SIZE}\n' \
                      f'num of epochs: {num_of_epochs}\n' \
                      f'learning rate: {learning_rate}\n' \
                      f'train elem spec:\n{train_ds_0_cqt.element_spec}\n'

    LOG.info(experiment_info)

    reverb_generator = ReverbGenerator(input_sequence_length=cqt_input_length,
                                       feature_dims=feature_dims,
                                       learning_rate=learning_rate,
                                       out_sequence_length=output_length,
                                       model_struct=model_struct)

    start_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    experiment_path = './experiments/reverb/' + start_time + '_experiment'
    os.mkdir(experiment_path)

    with open(f'{experiment_path}/properties.txt', 'w') as fp:
        reverb_generator.model.summary(print_fn=lambda x: fp.write(x + '\n'))
        fp.write('\n')
        fp.write(f'input length: {cqt_input_length}\n')
        fp.write(f'label length: {output_length}\n')
        fp.write(f'training epochs: {num_of_epochs}')
        fp.write(f'start time: {start_time}')
        fp.write(experiment_info)

    """tf.keras.utils.plot_model(
        reverb_generator.model,
        to_file=f"{experiment_path}/model.png",
        show_shapes=True,
        show_dtype=False,
        show_layer_names=False,
        rankdir="TB",
        expand_nested=False,
        dpi=120,
        layer_range=None,
        show_layer_activations=False,
    )"""

    LOG.info('Training model...\n')

    train_ds_0_cqt_batched = train_ds_0_cqt.batch(BATCH_SIZE)
    val_ds_0_cqt_batched = val_ds_0_cqt.batch(BATCH_SIZE)

    reverb_generator.train_model(train_ds_0_cqt_batched, val_ds_0_cqt_batched, epochs=num_of_epochs)

    # reverb_generator.model.save(f'{experiment_path}/model.h5')

    LOG.info('\n...Done. Generating Batch Predictions...')

    test_ds_0_batched = test_ds_0_cqt.batch(1)

    iteri = iter(test_ds_0_batched)
    test_batch = [next(iteri) for _ in range(int((cqt_out_len - input_ln) / output_length))]

    # print(len(test_batch))

    predict_reverb_for_batch_cqt(test_batch,
                                 reverb_generator,
                                 label_length=output_length,
                                 norm=norm_train,
                                 experiment_path=experiment_path,
                                 cqt_wind_size=cqt_window_length)

    LOG.info('...Done. Generating full predictions...')

    signal_starts, labelsignals = get_start_sequences_cqt(test_cqts_0_per_datapoint,
                                                          input_len=input_ln,
                                                          label_len=output_ln)

    predict_complete_reverb_cqt(signal_starts,
                                labelsignals,
                                model=reverb_generator,
                                input_length=input_ln,
                                label_length=output_ln,
                                cqt_window_size=cqt_window_length,
                                experiment_path=experiment_path,
                                norm=norm_train)

    LOG.info('...Done.')


def get_start_sequences_cqt(test_ds, input_len, label_len):
    amount_to_keep = input_len + label_len
    signal_starts = []
    labels = []
    print(f'amount_tokeep: {amount_to_keep}')

    for elem in test_ds:
        for signal in elem:
            labels.append(signal)
            # prep.draw_cqt(np.transpose(labels[-1][:amount_to_keep]), cqt_window_size)

            signal_start = signal[:input_len]
            # signal_start_ds = tf.data.Dataset.from_tensor_slices(([signal_starts[:input_len]], [signal_start[label_len:]]))
            signal_starts.append(signal_start)

    return signal_starts, labels


def predict_complete_reverb_cqt(inputs, labelsignals,
                                input_length, label_length,
                                experiment_path,
                                norm,
                                model=None,
                                cqt_window_size=512):
    predictions = []

    path = experiment_path + '/predictions'
    os.mkdir(path)

    inputs = inputs[:6]
    for index, signal in enumerate(inputs):

        predicted_cqts = np.transpose(signal)

        signal /= norm
        real = tf.expand_dims(np.real(signal), 1)
        img = tf.expand_dims(np.imag(signal), 1)

        input_signal = np.concatenate((real[:, 0], img[:, 0]), axis=-1)
        input_signal = np.expand_dims(input_signal, axis=0)
        # print(f'first input shape: {input_signal.shape}')

        label = np.transpose(labelsignals[index])
        # print(f'cqt_of_label: {label.shape}')
        end_sample_index = label.shape[-1]

        output_cqt_length = predicted_cqts.shape[-1]
        while output_cqt_length < end_sample_index:
            prediction = model.predict(input_signal)

            if np.isnan(prediction).any():
                break

            if label_length == input_length:
                input_signal = prediction
            elif label_length < input_length:
                input_signal = input_signal[:, int(label_length):]
                input_signal = tf.concat((input_signal, prediction), axis=1)
            else:
                input_signal = prediction[:, int(label_length - input_length):]

            prediction *= norm
            prediction_reshape = tf.reshape(prediction, [label_length, int(prediction.shape[-1] / 2), 2])
            prediction_cqt = np.transpose(prep.make_complex(prediction_reshape))
            predictions.append(prediction_cqt)
            predicted_cqts = np.concatenate((predicted_cqts, prediction_cqt), axis=-1)
            output_cqt_length = predicted_cqts.shape[-1]

        # print(f'cqt_of_pred: {predicted_cqts.shape}')

        # CQT
        prep.draw_cqt(predicted_cqts, save=True, filename=f'{index}_prediction', path=path, hop_length=cqt_window_size)
        prep.draw_cqt(label, save=True, filename=f'{index}_label', path=path, hop_length=cqt_window_size)

        if label.shape[-1] != predicted_cqts.shape[-1]:
            print(f'labelshaoe: {label.shape}')
            print(f'predshed: {predicted_cqts.shape}')
            predicted_cqts = predicted_cqts[:, :label.shape[-1]]

        cqt_diff = abs(abs(label) - abs(predicted_cqts))
        prep.draw_cqt(cqt_diff, save=True, filename=f'{index}_difference', path=path, hop_length=cqt_window_size)

        prep.draw_multiple_cqts(np.concatenate(
            (np.expand_dims(label, axis=0), np.expand_dims(predicted_cqts, axis=0), np.expand_dims(cqt_diff, axis=0)),
            axis=0),
            hop_length=cqt_window_size,
            plotnames=['label', 'prediction', 'difference'],
            filename=f'{index}_CQTs',
            path=path,
            save=True)

        # Audio
        label_as_audio = prep.signal_from_cqts(label, hop_length=cqt_window_size)
        pred_as_audio = prep.signal_from_cqts(predicted_cqts, hop_length=cqt_window_size)

        diff_as_audio = label_as_audio - pred_as_audio

        write(f'{path}/{index}_signal_predicted.wav', 48000, pred_as_audio)
        write(f'{path}/{index}_signal_label.wav', 48000, label_as_audio)

        draw_multiple_plots(np.concatenate(
            (np.expand_dims(
                label_as_audio, axis=0), np.expand_dims(pred_as_audio, axis=0), np.expand_dims(diff_as_audio, axis=0)),
            axis=0),
            save=True,
            path=path,
            filename=f'{index}_Audio',
            limit=1.2,
            plotnames=['label', 'prediction'])


def predict_reverb_for_batch_cqt(batch, model, label_length, norm, experiment_path, cqt_wind_size):
    labels_combine = []
    predictions = []

    path = experiment_path + '/batchpredictions'
    os.mkdir(path)

    first_input = batch[0][0] * norm
    first_input = np.reshape(first_input, [first_input.shape[1], int(first_input.shape[-1] / 2), 2])
    first_input = prep.make_complex(first_input)
    first_input = np.transpose(first_input)

    labels_combine.append(first_input)
    predictions.append(first_input)

    for index, signal in enumerate(batch):
        input_signal = signal[0]
        labels = signal[1]

        # labels
        labels *= norm
        labels = tf.reshape(labels, [-1, label_length, int(labels.shape[-1] / 2), 2])

        labels_complex = list(map(prep.make_complex, labels))
        labels_complex = np.transpose(np.concatenate(labels_complex, axis=0))
        labels_combine.append(labels_complex)
        # print(labels_complex.shape)

        # predictions
        prediction = model.predict(input_signal)

        prediction *= norm
        prediction_reshape = tf.reshape(prediction, [-1, label_length, int(prediction.shape[-1] / 2), 2])

        prediction_complex = list(map(prep.make_complex, prediction_reshape))
        prediction_complex = np.transpose(np.concatenate(prediction_complex, axis=0))
        predictions.append(prediction_complex)

    labels_concat = np.concatenate(labels_combine, axis=1)
    print(f'labels cqts shape: {labels_concat.shape}')
    prep.draw_cqt(labels_concat, save=True, filename='cqt_labels', path=path, hop_length=cqt_wind_size)

    predict_concat = np.concatenate(predictions, axis=1)
    print(f'predicted cqts shape: {predict_concat.shape}')
    prep.draw_cqt(predict_concat, save=True, filename='cqt_predictions', path=path, hop_length=cqt_wind_size)

    cqt_diff = abs(abs(labels_concat) - abs(predict_concat))
    prep.draw_cqt(cqt_diff, save=True, filename='cqt_diff', path=path, hop_length=cqt_wind_size)

    prep.draw_multiple_cqts(np.concatenate(
        (np.expand_dims(labels_concat, axis=0), np.expand_dims(predict_concat, axis=0),
         np.expand_dims(cqt_diff, axis=0)),
        axis=0),
        hop_length=cqt_wind_size,
        plotnames=['label', 'prediction', 'difference'],
        filename=f'CQTs',
        path=path,
        save=True)

    label_audio = prep.signal_from_cqts(labels_concat, cqt_wind_size)
    draw_plot(label_audio,
              title='label as audio',
              save=True,
              filename='audio_labels',
              path=path,
              limit=1.0
              )

    pred_audio = prep.signal_from_cqts(predict_concat, cqt_wind_size)
    draw_plot(pred_audio,
              title='pred as audio',
              save=True,
              filename='audio_pred',
              path=path,
              limit=1.0
              )

    diff_as_audio = label_audio - pred_audio
    draw_plot(diff_as_audio,
              title='diff as audio',
              save=True,
              filename='audio_diff',
              path=path,
              limit=1.0
              )

    draw_multiple_plots(np.concatenate(
        (np.expand_dims(
            label_audio, axis=0), np.expand_dims(pred_audio, axis=0), np.expand_dims(diff_as_audio, axis=0)),
        axis=0),
        save=True,
        path=path,
        filename=f'Audio',
        limit=1.2,
        plotnames=['label', 'prediction', 'difference'])

    write(f'{path}/prediction.wav', 48000, pred_audio)
    write(f'{path}/labels.wav', 48000, label_audio)


def initialize_log(log):
    global LOG
    LOG = log
