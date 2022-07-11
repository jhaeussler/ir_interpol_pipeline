import numpy as np
import pandas as pd
import pickle
import os
import mat73
from scipy.io import loadmat
from dataclasses import dataclass, field
from multiprocessing import Pool

from src.preprocessing import normalize_audio_signal_to_one

global LOG


# This is dependent on the underlying structure of the files you want to load. Consider this the interface for the
# data on the hard disk, that should be loaded into IRDataPoint-Instances.
# The returned Dict must include all the keys that the DataClasses are using.
def load_matlab_file_old(filename, first_file_in_dataset=False):
    """
    This function takes the filename of a .mat-file and loads all the data from that file into a dictionary.
    The Dictionary is then returned.

    The returned Dict must include the following keys:
        'MicPos': (X, Y, Z) position of microfon array origin in room coordinates
        'sweepRec': Multichannel Recording of the Sweep signal with Room Response, one channel per mic in array
        'irs': Impulse Responses generated from the sweepRec-signals

        'author': Person who recorded this Dataset
        'room': The recorded Room represented by this Data Set
        'info': Go wild
        'speakers': identity of used speakers
        'sample_f': Sampling Frequency
        'GridOff': (X, Y, Z)-Offset coordinates of recording Grid to Room Coordinates
        'sweepLoud': The loudness of the Sweep Signal at recording time(s)
        'sweep': The original (mono) Sweep-Signal that is played over the loudspeakers in the room
    """

    data_irs = loadmat(filename)['irs'][0][0]

    flattened_data = {}
    flattened_data.update({'MicPos': (data_irs[4].flat[0], data_irs[5].flat[0], 0),
                           'sweepRec': [np.transpose(data_irs[8])],
                           # Convert irs to numpy array and transpose it so the rows correspond to the audio signals,
                           # not the columns
                           'irs': [np.transpose(data_irs[11])]})

    if first_file_in_dataset:
        # generate the fields a new IRDataSet instance as well
        flattened_data.update({'author': str(data_irs[0].flat[0]),
                               'room': str(data_irs[1].flat[0]),
                               'info': str(data_irs[2].flat[0]),
                               'speakers': tuple('A'),
                               'sample_f': int(data_irs[6].flat[0]),
                               'GridOff': (data_irs[9].flat[0].flat[0], data_irs[10].flat[0].flat[0], 0),
                               'sweepLoud': float(data_irs[7].flat[0][0].flat[0]),
                               'sweep': data_irs[7].flat[0][1]})

    return flattened_data


def load_file_matlab_v73(filename, first_file_in_dataset=False):
    flattened_data = {}

    data_dict = mat73.loadmat(filename)

    flattened_data.update({'MicPos': tuple(data_dict['irs']['micPos']),
                           'sweepRec': [np.transpose(array) for array in data_dict['irs']['sweepRec']],
                           'irs': [np.transpose(array) for array in data_dict['irs']['ir']]
                           })

    if first_file_in_dataset:
        flattened_data.update({'author': data_dict['irs']['author'],
                               'room': data_dict['irs']['room'],
                               'info': data_dict['irs']['info'],
                               'speakers': tuple(data_dict['irs']['speakerNames']),
                               'speaker_positions': tuple(tuple(array) for array in data_dict['irs']['speakerPos']),
                               'speaker_type': data_dict['irs']['speakerType'],
                               'speaker_az_el': tuple(tuple(array) for array in data_dict['irs']['speakerAzEl']),
                               'sample_f': int(data_dict['irs']['fs']),
                               'GridOff': tuple(data_dict['irs']['gridOffXYZ']),
                               'sweepLoud': float(data_dict['irs']['sweepLoudness']),
                               'sweep': data_dict['irs']['sweep']
                               })

    return flattened_data


def preprocess_audio(flattened_data):
    # Pass all the signals in 'irs' to a function that searches the signals for sudden changes in energy and returns an
    # appropriate index. These changes would occur where the first direct sound reached the microphone.
    # Also save the max value of any signal in this datapoint in a separate variable, for later normalization
    max_vol = 0.
    audios_normalized_to_one = []
    index = 0

    for loudspeaker_recordings in flattened_data['irs']:
        irs_normalized_to_one = []

        for signal in loudspeaker_recordings:
            irs_normalized_to_one.append(normalize_audio_signal_to_one(signal))

            if max(signal) > max_vol:
                max_vol = max(signal)

        audios_normalized_to_one.append(irs_normalized_to_one)
        index += 1
    return max_vol, audios_normalized_to_one


def load_datapoint(flattened_data):
    """
    Load a single Data Point from python Dictionary to an existing dataset.
    Note that this function doesn't accept a filename as param, the load_file()-function has to be called first
    and then it's output has to be handed to this function. This is just makes it easier to handle different data sets.
    This function does not need to be changed. Changes in Data Structures of loaded Data Sets can and should be managed
    in the load_file-func
    """
    # print(flattened_data['MicPos'])
    max_vol, audios_normalized_to_one = preprocess_audio(flattened_data)

    return pd.DataFrame(
        {'MicPos': [flattened_data['MicPos']],
         'sweepRec': [flattened_data['sweepRec']],
         'irs': [flattened_data['irs']],
         'max_ir_vol_value': [max_vol],
         'irs_normalized_to_entire_dataset': [[]],
         'irs_normalized_to_one': [np.array(audios_normalized_to_one)]
         })


@dataclass
class IRDataSet:
    """This is a Wrapper Class for the datasets consisting of multiple IRDataInstances."""
    author: str
    room: str
    info: str
    speakers: list
    f_sample: int
    GridOff: tuple
    sweepLoudness: float
    sweep: np.array
    data: pd.DataFrame = field(default_factory=lambda: pd.DataFrame())

    # This is usually not good practice but I want to be able to pretty-print the data sets. This should be a more
    # JSON-dictionary kinda styled string holding all info to reconstruct class instances
    def __repr__(self):
        return f'\nDataset:\n' \
               f'Author: {self.author}\n' \
               f'Room: {self.room}\n' \
               f'Info: {self.info}\n' \
               f'Loudspeakers: {self.speakers}\n' \
               f'samplingFS: {self.f_sample}\n' \
               f'Offset of the Grid in the recorded Room : {self.GridOff}\n' \
               f'Sweep:(\n' \
               f'   loudness: {self.sweepLoudness}\n' \
               f'   in_signal_length: {len(self.sweep)})\n\n' \
               f'Amount of Data Points: {len(self.data)}\n' \
               f'Columns:\n' \
               f'{self.data.columns}\n'

    def __len__(self):
        return len(self.data)

    @classmethod
    def create_new_dataset(cls, filename, load_old=False):
        """
        Create a new Dataset from file. The load_file()-function is called with first=True to load additional info from
        file for the dataset creation.
        """
        if load_old:
            flattened_data = load_matlab_file_old(filename, True)
        else:
            flattened_data = load_file_matlab_v73(filename, True)
        # call class constructor with parameters generated from load_file-func
        return cls(author=flattened_data['author'],
                   room=flattened_data['room'],
                   info=flattened_data['info'],
                   speakers=flattened_data['speakers'],
                   f_sample=flattened_data['sample_f'],
                   GridOff=flattened_data['GridOff'],
                   sweepLoudness=flattened_data['sweepLoud'],
                   sweep=flattened_data['sweep'],
                   data=load_datapoint(flattened_data)
                   )

    def sort_dataset(self):
        LOG.info('Sorting Data Points...')
        self.data = self.data.sort_values(by='MicPos', key=lambda col: col.map(lambda pos: pos[0] * 10 + pos[1]))
        self.data.index = np.arange(len(self.data))

    def load_sdm(self, sdm_path):
        data = loadmat(sdm_path)
        sdm = data['sdm_data'][0]

        sdm = unpack_sdm_dicts(
            list(map(unpack_sdm_to_dict, sdm)),
            num_of_loudspeakers=len(self.speakers))

        sdm.sort(key=lambda entry: entry[0]['RecPos'][0] * 10 + entry[0]['RecPos'][1])

        sdm = pd.DataFrame(sdm)
        sdm.columns = [f'sdm_speaker{index + 1}' for index in range(len(sdm.columns))]

        self.data = pd.concat([self.data, sdm], axis=1)
        LOG.info('SDM loaded.')


def unpack_sdm_dicts(sdm_dicts, num_of_loudspeakers):
    zaehler = 0
    datapoints = []
    for elem in sdm_dicts:
        if zaehler % num_of_loudspeakers == 0:
            datapoints.append([elem])
        else:
            datapoints[-1].append(elem)
        zaehler += 1

    return datapoints


def unpack_sdm_to_dict(sdm_packed):
    sdm_as_dict = {}
    sdm_as_dict.update({
        'sdm': sdm_packed[0][:72000][:],
        'sdm_ir': np.transpose(sdm_packed[1])[6][:72000],
        'LS': sdm_packed[2][0][0],
        'RecPos': sdm_packed[3][0]
    })

    return sdm_as_dict


def create_dataset_from_files(path, sdm_path=None, load_old=False):
    """
    Starts the process of loading a new dataset from files.
    """

    LOG.info('Loading Data Set...')

    entries = os.listdir(path)
    paths_and_filenames = [(path, str(entry)) for entry in entries]

    first_entry = paths_and_filenames.pop()
    dataset = IRDataSet.create_new_dataset(os.path.join(first_entry[0], first_entry[1]), load_old=load_old)
    LOG.info("Dataset created. First data point loaded.")

    with Pool(os.cpu_count()) as multiproc_pool:
        if load_old:
            datapoints = list(multiproc_pool.map(load_rest_of_dataset_old_matlab, paths_and_filenames))
        else:
            datapoints = list(multiproc_pool.map(load_rest_of_dataset_matlab_73, paths_and_filenames))

    dataset.data = pd.concat([dataset.data, pd.concat(datapoints)])
    dataset.sort_dataset()

    if sdm_path is not None:
        dataset.load_sdm(sdm_path)

    LOG.info(f'Loaded dataset with {len(dataset.data)} datapoints.')
    LOG.info('...Done.')

    LOG.info(dataset)
    # Save new dataset as .pkl to disk
    save_dataset_to_disk(dataset)


def load_rest_of_dataset_matlab_73(path_and_filename):
    if path_and_filename[1].endswith(".mat"):
        LOG.info("Loading next Datapoint...")
        return load_datapoint(load_file_matlab_v73(os.path.join(path_and_filename[0], path_and_filename[1])))


def load_rest_of_dataset_old_matlab(path_and_filename):
    if path_and_filename[1].endswith(".mat"):
        LOG.info("Loading next Datapoint...")
        return load_datapoint(load_matlab_file_old(os.path.join(path_and_filename[0], path_and_filename[1])))


def save_dataset_to_disk(dataset):
    """
    What do you know.. this saves the given dataset as a binary pkl-file.. wild.

    :param dataset: input data set that is to be saved to disk
    :return: void
    """
    filename = f'./pickle/{dataset.room.replace(" ", "_")}' \
               f'_{dataset.info.replace(" ", "_")}_{dataset.author.replace(" ", "_")}.pkl'

    LOG.info('Saving to picklefile. This may take a while...')

    if not os.path.isdir("./pickle"):
        os.mkdir("./pickle")
        LOG.info("created dir './pickle'")
    else:
        LOG.info("Dir './pickle' already exists. Dumping pickle-file...")

    with open(filename, 'wb') as fp:
        pickle.dump(dataset, fp)
        LOG.info('...Saved.')


def load_dataset_from_disk(filepath):
    """
    Load a dataset from binary pkl-file.

    :param filepath: path to file containing the dataset
    :return: the loaded dataset
    """
    if os.path.isfile(filepath):
        LOG.info(f'Loading DataSet {filepath}')
        with open(filepath, 'rb') as fp:
            return pickle.load(fp)
    else:
        LOG.info(f'Dataset {filepath} was not found. Moving on.')
        return False


def initialize_log(log):
    global LOG
    LOG = log
