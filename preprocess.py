import midi_stuff
import transpose
import os
import numpy as np
import pickle
import shutil

def process(directory="midis"):
    folder = './data2'
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

    current_dir = os.getcwd()
    # transpose.transpose(directory)
    os.chdir(current_dir)
    vocab_temp = midi_stuff.convert_all(directory, midi_stuff.convert_to_matrix)

    try:
        with open("matrix_vocab.pkl", "rb") as f:
            vocab = pickle.load(f)
    except FileNotFoundError:
        vocab = [-1]
    if vocab_temp is not None:
        for v in vocab_temp:
            if v not in vocab:
                vocab.append(v)

    with open("matrix_vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)

    print("Finished preprocessing")

if __name__ == "__main__":
    process()