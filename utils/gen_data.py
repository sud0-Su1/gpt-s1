from utils.data_utils.wrangle_data import encode_to_integers
import os
import numpy as np

def train_valid_split(data, valid_size=0.1):
    """
    Split data into train and validation sets.

    :param data: The data to be split.
    :param valid_size: The size of the validation set.

    :return: The train and validation sets.
    """
    n = len(data)
    train_data = data[:int(n * (1 - valid_size))]
    valid_data = data[int(n * (1 - valid_size)):]
    return train_data, valid_data


def encode_train_valid(train_data, valid_data, char_to_int):
    """
    Encode the training and validation data.

    :param train_data: The training data to be encoded.
    :param valid_data: The validation data to be encoded.

    :return: The encoded training and validation data.
    """
    # Encode data
    train_data_encoded_id = encode_to_integers(char_to_int, train_data)
    valid_data_encoded_id = encode_to_integers(char_to_int, valid_data)

    # Store the encoded data
    train_data_encoded_id = np.array(train_data_encoded_id, dtype=np.int64)
    valid_data_encoded_id = np.array(valid_data_encoded_id, dtype=np.int64)
    train_data_encoded_id.tofile(os.path.join('data', 'train_data_encoded_id.bin'))
    valid_data_encoded_id.tofile(os.path.join('data', 'valid_data_encoded_id.bin'))
