import os
from typing import Union

import mido

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
import numpy as np
import shutil
from math import log
import pickle
import ast

from AiInterface import AiInterface

try:
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
except:
    print("No GPU")

class Ai1(AiInterface):
    # ToDo :: add vocab functionality
    def midi_to_data(self, midi: mido.MidiFile, vocabs: list) -> (np.array, list):
        mid = mido.MidiFile(midi)
        ticks_per_beat = mid.ticks_per_beat
        simple = []
        simple2 = []
        offset = 0

        for i, msg in enumerate(mido.merge_tracks(mid.tracks)):
            if msg.type[:4] == "note":
                note = msg.note
                vel = msg.velocity
                time = msg.time / ticks_per_beat + offset

                if vel != 0:
                    simple2.append([note, vel, time - offset])
                    simple.append([note, vel, time, 0])
                    offset = 0

                else:
                    offset = time
                    ind = len(simple) - 1
                    length = time
                    # Loop through end of list until all note with current node value's length is set
                    while ind >= 0:
                        if simple[ind][0] == note:
                            if simple[ind][3] == 0:
                                simple[ind][3] = length
                            else:
                                break

                        time = simple[ind][2]
                        length += time
                        ind -= 1

        data = [str(x) for x in simple]

        return np.array(list(map(lambda x: ast.literal_eval(x), data)))

    def data_to_midi_sequence(self, sequence: list) -> list:
        pass

    def get_dataset(self) -> Union[np.array, tf.data.Dataset]:
        pass

    def build_model(self, batch_size) -> keras.Model:
        pass

    def train(self, epochs=10, cont=False, lr=0.001, checkpoint_num=None):
        pass

    def generate_text(self, model, num, start, temperature) -> list:
        pass

    def guess(self, num, start, temp, model, checkpoint_num) -> list:
        pass