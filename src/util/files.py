import pickle


def load_data(file_path):
    file = open(file_path, 'rb')

    return pickle.load(file)
