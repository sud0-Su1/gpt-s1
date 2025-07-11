from utils.data_utils.download_data import get_data
from utils.data_utils.wrangle_data import char_to_integers, integers_to_char
from utils.data_utils.generate_data import train_valid_split, encode_train_valid

def main():
    """
    Entry point for the data generation
    Perform the following:
    1. Get the data
    2. Split the data into train and validation sets
    3. Encode the training and validation data
    """
    # Get the data
    data = get_data()

    # Split the data into train and validation sets
    train_data, valid_data = train_valid_split(data)

    chars = sorted(list(set(data)))

    # Encode the training and validation data
    char_to_int = char_to_integers(chars)
    integers_to_char(chars)
    encode_train_valid(train_data, valid_data, char_to_int)


if __name__ == "__main__":
    main()
