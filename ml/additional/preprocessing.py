from enum import Enum
import pandas as pd
import random
from ml.additional.utils import *
import h5py


class AirSimProp(Enum):
    Steering = "Steering"
    Throttle = "Throttle"
    Brake = "Brake"
    Speed = "Speed"
    Image = "ImageFile"
    # Old headers
    # Speed = "Speed (kmph)"
    # Image = "ImageName"


class PreProcess():
    AIRSIM_TEXT_FILE = "airsim_rec.txt"
    AIRSIM_IMAGE_FOLDER = "images"
    TRAINING_H5_FILE_NAME = "train.h5"
    VALIDATION_H5_FILE_NAME = "val.h5"
    H5_PROPERTY_LABEL = "label"
    H5_PROPERTY_PREVIOUS_STATE = "previous_state"
    H5_PROPERTY_IMAGE = "image"
    output_directory = None
    original_data = None
    data_mapping = None
    train_data = None
    validation_data = None
    batch_size = 32
    split_ratio = (0.7, 0.3)

    #function for testing
    def prepare_test_data(self, input_folders):
        self.original_data = {}
        for folder in input_folders:
            print('Reading data from {0}...'.format(folder))
            current_df = pd.read_csv(os.path.join(folder, self.AIRSIM_TEXT_FILE), sep='\t')
            # current_df = pd.read_csv(os.path.join(folder, self.AIRSIM_TEXT_FILE))
            for i in range(1, current_df.shape[0] - 1, 1):
                previous_state = list(current_df.iloc[i - 1][[AirSimProp.Steering.value, AirSimProp.Throttle.value,
                                                              AirSimProp.Brake.value, AirSimProp.Speed.value]])
                current_label = list((current_df.iloc[i][[AirSimProp.Steering.value]] + current_df.iloc[i - 1][[AirSimProp.Steering.value]] +
                                      current_df.iloc[i + 1][[AirSimProp.Steering.value]]) / 3.0)
                image_filepath = os.path.join(os.path.join(folder, self.AIRSIM_IMAGE_FOLDER), current_df.iloc[i][AirSimProp.Image.value]).replace(
                    '\\', '/')

                if not os.path.exists(image_filepath):
                    continue

                # Sanity check
                if (image_filepath in self.original_data):
                    print('Error: attempting to add image {0} twice.'.format(image_filepath))

                self.original_data[current_df.iloc[i][AirSimProp.Image.value]] = (current_label, previous_state)

    def get_test_data(self):
        return self.original_data

    # function for training
    def prepare_training_data(self, input_folders, output_directory, batch_size=None):
        """ Data map generator for simulator(AirSim) data. Reads the driving_log csv file and returns a list of 'center camera image name - label(s)' tuples
                       Inputs:
                           folders: list of folders to collect data from
                       Returns:
                           mappings: All data mappings as a dictionary. Key is the image filepath, the values are a 2-tuple:
                               0 -> label(s) as a list of double
                               1 -> previous state as a list of double
                """
        self.output_directory = output_directory
        if batch_size:
            self.batch_size = batch_size
        if self.check_exist_processing_files():
            return
        self.original_data = {}
        for folder in input_folders:
            print('Reading data from {0}...'.format(folder))
            current_df = pd.read_csv(os.path.join(folder, self.AIRSIM_TEXT_FILE), sep='\t')
            # current_df = pd.read_csv(os.path.join(folder, self.AIRSIM_TEXT_FILE))
            for i in range(1, current_df.shape[0] - 1, 1):
                previous_state = list(current_df.iloc[i - 1][[AirSimProp.Steering.value, AirSimProp.Throttle.value,
                                                              AirSimProp.Brake.value, AirSimProp.Speed.value]])
                current_label = list((current_df.iloc[i][[AirSimProp.Steering.value]] + current_df.iloc[i - 1][[AirSimProp.Steering.value]] +
                                      current_df.iloc[i + 1][[AirSimProp.Steering.value]]) / 3.0)
                image_filepath = os.path.join(os.path.join(folder, self.AIRSIM_IMAGE_FOLDER), current_df.iloc[i][AirSimProp.Image.value]).replace(
                    '\\', '/')

                # Sanity check
                if (image_filepath in self.original_data):
                    print('Error: attempting to add image {0} twice.'.format(image_filepath))

                self.original_data[image_filepath] = (current_label, previous_state)

        self.data_mapping = [(key, self.original_data[key]) for key in self.original_data]

        random.shuffle(self.data_mapping)

    def check_exist_processing_files(self):
        output_files = [os.path.join(self.output_directory, f) for f in [self.TRAINING_H5_FILE_NAME, self.VALIDATION_H5_FILE_NAME]]
        if any([os.path.isfile(f) for f in output_files]):
            print("Preprocessed data already exists at: {0}. Skipping preprocessing.".format(self.output_directory))
            return True
        return False

    def start_processing(self, train_val_split=None):
        """ Primary function for data pre-processing. Reads and saves all data as h5 files.
                Inputs:
                    folders: a list of all data folders
                    output_directory: location for saving h5 files
                    train_eval_test_split: dataset split ratio
        """
        if self.check_exist_processing_files():
            return

        if train_val_split:
            self.split_ratio = train_val_split
        self.split_train_validation_data()

        # Save data sets

        self.save_data_as_h5(self.train_data, os.path.join(self.output_directory, self.TRAINING_H5_FILE_NAME))
        self.save_data_as_h5(self.validation_data, os.path.join(self.output_directory, self.VALIDATION_H5_FILE_NAME))

    def split_train_validation_data(self):
        """Simple function to create train, validation and test splits on the data.
                Inputs:
                    all_data_mappings: mappings from the entire dataset
                    split_ratio: (train, validation, test) split ratio
                Returns:
                    train_data_mappings: mappings for training data
                    validation_data_mappings: mappings for validation data
                    test_data_mappings: mappings for test data
        """
        if round(sum(self.split_ratio), 5) != 1.0:
            raise Exception("Error: Your splitting ratio should add up to 1")

        train_split = int(len(self.data_mapping) * self.split_ratio[0])

        self.train_data = self.data_mapping[0:train_split]
        self.validation_data = self.data_mapping[train_split:]

    def generator_for_h5py(self, data_mappings):
        """
        This function batches the data for saving to the H5 file
        """
        for chunk_id in range(0, len(data_mappings), self.batch_size):
            # Data is expected to be a dict of <image: (label, previousious_state)>
            # Extract the parts
            data_chunk = data_mappings[chunk_id:chunk_id + self.batch_size]
            if (len(data_chunk) == self.batch_size):
                image_names_chunk = [a for (a, b) in data_chunk]
                labels_chunk = np.asarray([b[0] for (a, b) in data_chunk])
                previous_state_chunk = np.asarray([b[1] for (a, b) in data_chunk])

                # Flatten and yield as tuple
                yield (image_names_chunk, labels_chunk.astype(float), previous_state_chunk.astype(float))
                if chunk_id + self.batch_size > len(data_mappings):
                    break
                    # raise StopIteration
        # raise StopIteration

    def save_data_as_h5(self, data_mappings, target_file_path):
        """
        Saves H5 data to file
        """
        print('Processing {0}...'.format(target_file_path))
        gen = self.generator_for_h5py(data_mappings)

        image_names_chunk, labels_chunk, previous_state_chunk = next(gen)
        images_chunk = np.asarray(read_images_from_path(image_names_chunk))
        row_count = images_chunk.shape[0]

        check_and_create_dir(target_file_path)
        with h5py.File(target_file_path, 'w') as f:
            # Initialize a resizable dataset to hold the output
            images_chunk_maxshape = (None,) + images_chunk.shape[1:]
            labels_chunk_maxshape = (None,) + labels_chunk.shape[1:]
            previous_state_maxshape = (None,) + previous_state_chunk.shape[1:]
            dset_images = f.create_dataset(self.H5_PROPERTY_IMAGE, shape=images_chunk.shape, maxshape=images_chunk_maxshape,
                                           chunks=images_chunk.shape, dtype=images_chunk.dtype)

            dset_labels = f.create_dataset(self.H5_PROPERTY_LABEL, shape=labels_chunk.shape, maxshape=labels_chunk_maxshape,
                                           chunks=labels_chunk.shape, dtype=labels_chunk.dtype)

            dset_previous_state = f.create_dataset(self.H5_PROPERTY_PREVIOUS_STATE, shape=previous_state_chunk.shape,
                                                   maxshape=previous_state_maxshape,
                                                   chunks=previous_state_chunk.shape, dtype=previous_state_chunk.dtype)

            dset_images[:] = images_chunk
            dset_labels[:] = labels_chunk
            dset_previous_state[:] = previous_state_chunk

            for image_names_chunk, label_chunk, previous_state_chunk in gen:
                image_chunk = np.asarray(read_images_from_path(image_names_chunk))

                # Resize the dataset to accommodate the next chunk of rows
                dset_images.resize(row_count + image_chunk.shape[0], axis=0)
                dset_labels.resize(row_count + label_chunk.shape[0], axis=0)
                dset_previous_state.resize(row_count + previous_state_chunk.shape[0], axis=0)
                # Write the next chunk
                dset_images[row_count:] = image_chunk
                dset_labels[row_count:] = label_chunk
                dset_previous_state[row_count:] = previous_state_chunk

                # Increment the row count
                row_count += image_chunk.shape[0]
        print('Finished saving {0}.'.format(target_file_path))


