from math import pi
import os
import requests
import pickle

def get_data():
    """
    Downloads the data from the tinyshakespeare dataset if it is not already present.

    :return: The text data from the tinyshakespeare dataset.
    """
    input_file_path = os.path.join('data', 'tinyshakespeare.txt')
    if not os.path.exists(input_file_path):
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        with open(input_file_path, 'w') as f:
            f.write(requests.get(url).text)

    with open(input_file_path, 'r', encoding='utf-8') as f:
        data = f.read()
    
    # Save the vocab size
    chars = sorted(list(set(data)))
    pickle.dump({ 'vocab_size' : len(chars) }, open(os.path.join('data', 'vocab_size.pkl'), 'wb'))

    return data
