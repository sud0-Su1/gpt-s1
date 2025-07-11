import os
import pickle

def char_to_integers(chars):
    """
    Pickle the mapping of characters to integers

    :param chars: chars to encode

    :return: a dictionary of characters to integers
    """
    char_to_int = dict((c, i) for i, c in enumerate(chars))
    
    with open(os.path.join('data', 'char_to_int.pkl'), 'wb') as f:
        pickle.dump(char_to_int, f)
    
    return char_to_int


def integers_to_char(chars):
    """
    Pickle the mapping of integers to characters

    :param chars: chars to encode

    :return: a dictionary of integers to characters
    """
    int_to_char = dict((i, c) for i, c in enumerate(chars))
    
    with open(os.path.join('data', 'int_to_char.pkl'), 'wb') as f:
        pickle.dump(int_to_char, f)
    
    return int_to_char


def encode_to_integers(dict, s):
    """
    Encodes strings to integers

    :param s: string to encode

    :return: a list of integers
    """
    return [dict[c] for c in s]

def decode_to_string(dict, list_integers):
    """
    Decodes integers to strings

    :param list_integers: list of integers to decode

    :return: a string
    """
    return ''.join([dict[i] for i in list_integers])
