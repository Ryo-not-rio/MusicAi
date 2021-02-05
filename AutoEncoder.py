import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
import numpy as np
import shutil
import time
from typing import Union
import mido
import random
import pickle

from basicAiInterface import Interface


try:
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
except:
    print("No GPU")


BATCH_SIZE = 64
SEQ_LENGTH = 10
ENCODER_DIM = 64

class Ai(Interface):
    def __init__(self):
        self.vocab_file = "encoder.pkl"
        super().__init__("encoder_checkpoints", "encoder", BATCH_SIZE)
        self.loss = -1
        self.prev_loss = 100


        try:
            with open(self.vocab_file, "rb") as f:
                self.vocabs = pickle.load(f)
        except FileNotFoundError:
            self.process_all()  # Sets self.vocabs

    def midi_to_data(self, midi: mido.MidiFile, vocabs: list) -> (np.array, list):
        if not vocabs:
            vocabs = [[-1], [-1], [-1, 0]]
        ticks_per_beat = midi.ticks_per_beat
        simple_seq = [[-1, -1, -1]]
        offset = 0

        for i, msg in enumerate(mido.merge_tracks(midi.tracks)):
            if msg.type[:4] == "note":
                note = msg.note
                vel = msg.velocity
                time = msg.time / ticks_per_beat + offset

                if vel != 0:
                    if note not in vocabs[0]:
                        vocabs[0].append(note)

                    time = round(time, 6)
                    if time not in vocabs[1]:
                        vocabs[1].append(time)
                    simple_seq.append([note, time, 0])
                    offset = 0

                else:
                    offset = time
                    ind = len(simple_seq) - 1
                    length = time
                    # Loop through end of list until all note with current node value's length is set
                    change_ind = None
                    last_length = None

                    def update_length(last_length):
                        length = last_length
                        length = round(length, 5)
                        if length not in vocabs[2]:
                            vocabs[2].append(length)
                        simple_seq[change_ind][2] = length

                    while ind >= 0:
                        if simple_seq[ind][0] == note:
                            if simple_seq[ind][2] == 0:
                                change_ind = ind
                                last_length = length
                            elif change_ind is not None:
                                update_length(last_length)
                                break

                        elif ind == 0 and change_ind is not None:
                            update_length(last_length)
                            break

                        time = simple_seq[ind][1]
                        length += time
                        ind -= 1

        matrix_seq = [[-1] * 128]
        for i, event in enumerate(simple_seq):
            note, time, length = event
            next_matrix = matrix_seq[-1][:]
            next_matrix = [x - time if x - time > 0 else 0 for x in next_matrix]
            next_matrix[note] = max(next_matrix[note], length)
            if i > 0:
                matrix_seq.append(next_matrix)

        for i, event in enumerate(simple_seq):
            simple_seq[i] = [vocabs[j].index(x) for j, x in enumerate(event)]

        return [np.array(matrix_seq), np.array(simple_seq)], vocabs

    def process_all(self, midi_dir: str = "midis") -> list:
        print("Processing midis...")

        start = time.time()
        shutil.rmtree(self.data_dir)
        os.mkdir(self.data_dir)
        try:
            with open(self.vocab_file, "rb") as f:
                vocabs = pickle.load(f)
        except FileNotFoundError:
            vocabs = []

        for file in os.listdir(midi_dir):
            mid = mido.MidiFile(os.path.join(midi_dir, file))
            data, vocabs = self.midi_to_data(mid, vocabs)

            with open(os.path.join(self.data_dir, os.path.split(file)[-1][:-3] + "npz"), "wb") as f:
                np.savez_compressed(f, data[0], data[1])

        with open(self.vocab_file, "wb") as f:
            pickle.dump(vocabs, f)

        self.vocabs = vocabs
        print("processed all midi files.")
        print("time_taken: ", time.time() - start)
        return vocabs

    def get_dataset(self) -> Union[np.array, tf.data.Dataset]:
        def parse(array):
            if array.shape[0] < SEQ_LENGTH:
                return [np.concatenate((array, np.tile([0]*array.shape[-1], (SEQ_LENGTH-array.shape[0], 1))), axis=0)]
            else:
                return_list = []
                for i in range(array.shape[0]//SEQ_LENGTH):
                    return_list.append(array[i*SEQ_LENGTH:i*SEQ_LENGTH+SEQ_LENGTH])
                return return_list

        X = []
        y = []

        files = os.listdir(self.data_dir)
        for i in range(2):
            random.shuffle(files)
            for file in files:
                file_name = os.fsdecode(file)
                with(open(os.path.join(self.data_dir, file_name), "rb")) as f:
                    loaded = np.load(f)
                    data1, data2 = loaded['arr_0'], loaded['arr_1']
                    X += parse(data1)
                    y += parse(data2)

        X = np.array(X)
        dataset = tf.data.Dataset.from_tensor_slices((X, y))
        return dataset

    def build_encoder(self, batch_size):
        model = keras.Sequential([
            keras.layers.Input(batch_input_shape=(batch_size, SEQ_LENGTH, 128)),
            keras.layers.Flatten(),
            keras.layers.Dense(ENCODER_DIM, activation="relu")
        ])
        return model

    def build_decoder(self, batch_size):
        model = keras.Sequential([
            keras.layers.Input(batch_input_shape=(batch_size, ENCODER_DIM)),
            keras.layers.Dense(SEQ_LENGTH*3*len(self.vocabs[2]), activation="tanh"),
            keras.layers.Reshape((SEQ_LENGTH, 3, len(self.vocabs[2])))
        ])
        return model

    def build_model(self, batch_size=BATCH_SIZE):
        encoder = self.build_encoder(batch_size)
        decoder = self.build_decoder(batch_size)

        auto_encoder = keras.Model(encoder.input, decoder(encoder.output))

        return encoder, decoder, auto_encoder

    def train(self, epochs=10, cont=False, lr=0.001, checkpoint_num=None):
        dataset = self.get_dataset()
        train_data = dataset.skip(BATCH_SIZE).batch(BATCH_SIZE, drop_remainder=True)
        test_data = dataset.take(BATCH_SIZE).repeat(3).batch(BATCH_SIZE, drop_remainder=True)

        model = self.build_model()[2]

        # for d in train_data.take(1):
        #     print(d)
        #
        # exit()

        ini_epoch = 0
        if cont:
            latest = self.get_checkpoint(checkpoint_num)
            if latest is not None:
                ini_epoch = int(latest[len(self.check_dir)+6:])
                model.load_weights(latest)
        else:
            Interface.remove_checkpoints(self.check_dir)

        model.compile(optimizer="adam",
                      loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True, name="loss"),
                      metrics="accuracy",)
        checkpoint_call_back = tf.keras.callbacks.ModelCheckpoint(self.check_dir+"/ckpt_{epoch}", save_weights_only=True)

        model.fit(train_data,
                  epochs=epochs + ini_epoch, initial_epoch=ini_epoch,
                  callbacks=[checkpoint_call_back,
                             self.loss_callback(),
                             self.get_scheduler(lr, cont=cont),],
                  verbose=1,
                  validation_data=test_data,
                  validation_freq=3,
                  )

        return model

if __name__ == "__main__":
    ai = Ai()
    # ai.process_all()
    # converted = ai.midi_to_data(mido.MidiFile("midis/alb_esp1.mid"))

    # try_data = np.zeros((1, 200, 128), dtype=np.float32)
    # model = ai.build_model(1)
    # print(model(try_data))
    # prediction = ai.predict_vel(try_data, 1.0)

    # for d in ai.get_dataset().take(1):
    #     print(d)

    # ai.build_model()
    ai.train(1, cont=False)

