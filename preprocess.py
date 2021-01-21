import midi_stuff
import transpose
import os

def process(directory="midis"):
    current_dir = os.getcwd()
    # transpose.transpose(directory)
    os.chdir(current_dir)
    midi_stuff.convert_all(directory, midi_stuff.convert_to_matrix)

if __name__ == "__main__":
    process()