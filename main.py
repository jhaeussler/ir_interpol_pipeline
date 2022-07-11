import argparse
import os

from src.delux_logger import DeluxLogger, LogLevel
from src.handle_datasets import create_dataset_from_files, load_dataset_from_disk
from src.machine_learning_pipeline import generate_early_signal, generate_reverb, richtungsvektorexperiment, load_pretraind_models_and_predict_ir
from src.prepare_for_neural_nets import initialize_log as prepare_nn_log_init
from src.handle_datasets import initialize_log as hd_log_init
from src.machine_learning_pipeline import initialize_log as ml_pipe_log

# ----------------------------------
# Loading Parameters.
# ----------------------------------

arg_parser = argparse.ArgumentParser(description='Arguments for the ir_interpol module.',
                                     usage='main.py [-h] [--path_to_new_dataset]')
arg_parser.add_argument('--path_to_new_dataset', help='[only for creating new dataset] Provide path to new dataset')
arg_parser.add_argument('--model_struct', help='dense=0, conv=1, lstm=2', type=int, default=0)
arg_parser.add_argument('--training_epochs', help='Amount of training epochs', type=int, default=1)
arg_parser.add_argument('--sequence_length', help='length of prediction sequences', type=int, default=2)
arg_parser.add_argument('--output_sequence_length', help='length of NN output sequence', type=int, default=1)
arg_parser.add_argument('--cqt_window_length', help='length of transform window for cqt', type=int, default=2048)
arg_parser.add_argument('--learning_rate', help='learning rate of optimizer', type=float, default=1e-4)
arg_parser.add_argument('--run_early', help='run the early ir synthesis experiment', type=bool, default=False)
arg_parser.add_argument('--run_reverb', help='run the reverb synthesis experiment', type=bool, default=False)
arg_parser.add_argument('--run_direc_vec', help='run the direction vector synthesis experiment', type=bool,
                        default=False)

command_line_args = arg_parser.parse_args()

MODEL_STRUCT = command_line_args.model_struct
NUM_OF_EPOCHS = command_line_args.training_epochs
SEQUENCE_LENGTH = command_line_args.sequence_length
OUTPUT_LENGTH = command_line_args.output_sequence_length
CQT_WINDOW_LENGTH = command_line_args.cqt_window_length
LEARNING_RATE = command_line_args.learning_rate

RUN_EARLY_IR = command_line_args.run_early
RUN_REVERB = command_line_args.run_reverb
RUN_DIREC_VEC = command_line_args.run_direc_vec

print(f'Run Early: {RUN_EARLY_IR}')
print(f'Run reverb: {RUN_REVERB}')

# Set this true if you want to load a new dataset from files, make sure the right path is provided below.
# Note that the datasets-list will not be created and the ml-pipeline will not be called if this is True.
CREATE_NEW_DATASET = False

# location of audio recordings as .mat-files -> Path to data for creation of a new Dataset.
# Usually the datasets will be loaded from pickle-binary-files to save time and prevent polluting the console
# A new DataSet must once be formatted and saved to pickle-file. This is where you put the path to the dataset.
DATASET_LISTENING_LAB = ("5x5_Grid", "sdm_data_HL_oneSpeaker.mat")
DATASET_HE_1539B = ("jonathan_H1539b", "sdm_data_jonathan_H1539b.mat")
DATASET_HE_2505 = ("HE-2505", "sdm_data_jonathan_H2505.mat")
DATASET_ML_2102 = ("jonathan_ML2102", "sdm_data_jonathan_ML2102.mat")

# choose one from above
dataset = DATASET_HE_1539B

PATH_TO_NEW_DATASET = "./data/" + dataset[0]
PATH_TO_SDM = "./data/sdm/" + dataset[1]

LOAD_OLD_MATLAB_FILE = False
if dataset == DATASET_LISTENING_LAB:
    LOAD_OLD_MATLAB_FILE = True

# Here are all the datasets listed that where already analysed and saved to disk.
PICKLE_PATH_DATASET_LISTENING_LAB = "./pickle/ListeningLab_SDM_Florian_Klein.pkl"
PICKLE_PATH_DATASET_HE_1539B = "./pickle/H1539b_SDM_Florian_Klein.pkl"
PICKLE_PATH_DATASET_HE_2505 = "./pickle/H2505_SDM_Florian_Klein.pkl"
PICKLE_PATH_DATASET_ML_2102 = "./pickle/ML2-102_SDM_Florian_Klein.pkl"

# Set to True to generate Extra Output
RUN_DEBUG_CODE = False

LOG = DeluxLogger()


# ----------------------------------
# This is were everything is controlled from.
# ----------------------------------

def main():
    """
    The Main Function. This just runs the subprocesses in the desired configuration.
    Also some argument handling may occur. Also Logging.
    """
    datasets = []

    if CREATE_NEW_DATASET:
        create_dataset_from_files(PATH_TO_NEW_DATASET, PATH_TO_SDM, LOAD_OLD_MATLAB_FILE)

    else:

        try:
            os.makedirs('./experiments/early_signal')
        except FileExistsError as err:
            LOG.err(err)
        try:
            os.makedirs('./experiments/reverb')
        except FileExistsError as err:
            LOG.err(err)

        # ----------------------------------
        # Choose the Dataset(s) to load here.
        # ----------------------------------

        load_listening_lab = True
        if load_listening_lab:
            dataset_listening_lab = load_dataset_from_disk(PICKLE_PATH_DATASET_LISTENING_LAB)
            log_dataset_loaded(dataset_listening_lab)
            datasets.append(dataset_listening_lab)

        # dataset_he_1539b = load_dataset_from_disk(PICKLE_PATH_DATASET_HE_1539B)
        # log_dataset_loaded(dataset_he_1539b)
        # datasets.append(dataset_he_1539b)

        # dataset_he_2505 = load_dataset_from_disk(PICKLE_PATH_DATASET_HE_2505)
        # log_dataset_loaded(dataset_he_2505)
        # datasets.append(dataset_he_2505)

        if not load_listening_lab:
            dataset_ml_2102 = load_dataset_from_disk(PICKLE_PATH_DATASET_ML_2102)
            log_dataset_loaded(dataset_ml_2102)
            datasets.append(dataset_ml_2102)

        # ----------------------------------
        # Choose the Experiment to run here.
        # ----------------------------------

        if RUN_EARLY_IR:
            generate_early_signal(datasets[-1],
                                  # sequence_length=SEQUENCE_LENGTH,
                                  model_struct=MODEL_STRUCT,
                                  num_of_epochs=NUM_OF_EPOCHS,
                                  learning_rate=LEARNING_RATE
                                  )
        if RUN_REVERB:
            generate_reverb(datasets[-1],
                            model_struct=MODEL_STRUCT,
                            num_of_epochs=NUM_OF_EPOCHS,
                            sequence_length=SEQUENCE_LENGTH,
                            output_length=OUTPUT_LENGTH,
                            learning_rate=LEARNING_RATE,
                            cqt_window_length=CQT_WINDOW_LENGTH
                            )
        global RUN_DIREC_VEC
        RUN_DIREC_VEC = False
        if RUN_DIREC_VEC:
            richtungsvektorexperiment(datasets[-1], model_struct=MODEL_STRUCT, num_of_epochs=NUM_OF_EPOCHS)

        # Load Trained Models and predict a RIR
        # load_pretraind_models_and_predict_ir()


def log_dataset_loaded(loaded_dataset):
    LOG.info(f'Dataset with {len(loaded_dataset.data)} recordings loaded.')


def initialize_logger(log):
    """
    This is probably not be best way to pass the logger instance around, but it works.
    """
    if RUN_DEBUG_CODE:
        log.set_level(LogLevel.DEBUG)
    else:
        log.set_level(LogLevel.INFO)

    # for now I always want to run debug messages.
    log.set_level(LogLevel.DEBUG)

    log.info(f'Loglevel set to {log.level}')

    hd_log_init(log)
    prepare_nn_log_init(log)
    ml_pipe_log(log)


if __name__ == '__main__':
    initialize_logger(LOG)
    main()
