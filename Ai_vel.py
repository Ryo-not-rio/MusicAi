import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
import numpy as np
import shutil
import time
import pickle
from typing import Union
import mido
import random

from AiInterface import AiInterface

try:
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
except:
    print("No GPU")


TICKS_PER_BEAT = 512
BATCH_SIZE = 128
SEQ_LENGTH = 200

class Ai(AiInterface):
    def __init__(self):
        super().__init__("vel_checkpoints", "vel_data3", "vel_vocab.pkl", BATCH_SIZE, TICKS_PER_BEAT)

    def midi_to_data(self, midi: mido.MidiFile, vocabs=None) -> (np.array, list):
        ticks_per_beat = midi.ticks_per_beat
        simple_seq = [[-1, -1, -1]]
        vels = [-1]
        offset = 0

        for i, msg in enumerate(mido.merge_tracks(midi.tracks)):
            if msg.type[:4] == "note":
                note = msg.note
                vel = msg.velocity
                time = msg.time / ticks_per_beat + offset

                if vel != 0:
                    time = round(time, 6)
                    simple_seq.append([note, time, 0])
                    vels.append(vel)
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
        for i, event in enumerate(simple_seq[:-1]):
            note, time, length = event
            next_matrix = matrix_seq[-1][:]
            next_matrix = [x-time if x-time > 0 else 0 for x in next_matrix]
            next_matrix[note] = max(next_matrix[note], length)
            matrix_seq.append(next_matrix)

        return [np.array(matrix_seq), np.array(vels)]

    def data_to_midi_sequence(self, sequence: list) -> list:
        raise Exception("This model does not support this operation")

    def process_all(self, midi_dir: str = "midis") -> None:
        print("Processing midis...")

        start = time.time()
        shutil.rmtree(self.data_dir)
        os.mkdir(self.data_dir)

        for file in os.listdir(midi_dir):
            mid = mido.MidiFile(os.path.join(midi_dir, file))
            data = self.midi_to_data(mid)

            with open(os.path.join(self.data_dir, os.path.split(file)[-1][:-3] + "npz"), "wb") as f:
                np.savez_compressed(f, data[0], data[1])

        with open(self.vocab_file, "wb") as f:
            pickle.dump([], f)

        print("processed all midi files.")
        print("time_taken: ", time.time()-start)

    def get_dataset(self) -> Union[np.array, tf.data.Dataset]:
        X, y = [], []

        files = os.listdir(self.data_dir)
        for i in range(3):
            random.shuffle(files)
            for file in files:
                file_name = os.fsdecode(file)
                with(open(os.path.join(self.data_dir, file_name), "rb")) as f:
                    loaded = np.load(f)
                    X.append(loaded['arr_0'])
                    y += list(loaded['arr_1'])

        X, y = np.vstack(X), np.array(y)
        print(f"X shape: {X.shape}, Num sequences: {X.shape[0]/SEQ_LENGTH}, Batches: {X.shape[0]/(SEQ_LENGTH*BATCH_SIZE)}")
        dataset = tf.data.Dataset.from_tensor_slices((X, y))
        return dataset

    def build_model(self, batch_size) -> keras.Model:
        inputs = keras.layers.Input(batch_shape=(batch_size, None, 128), batch_size=batch_size)
        y = inputs
        y = keras.layers.Dense(512, activation="tanh")(y)

        y = keras.layers.GRU(512, stateful=True, return_sequences=True, return_state=True)(y)
        y = keras.layers.GRU(512, stateful=True, return_sequences=True, return_state=False)(y)
        y = keras.layers.Dropout(0.4)(y)

        y_1 = keras.layers.Dense(127)(y)

        m = keras.Model(inputs=inputs, outputs=y_1)
        # m.summary()
        return m

    def train(self, epochs=10, cont=False, lr=0.001, checkpoint_num=None):
        dataset = self.get_dataset()
        dataset = dataset.batch(SEQ_LENGTH, drop_remainder=True)
        train_data = dataset.skip(BATCH_SIZE).batch(BATCH_SIZE, drop_remainder=True)#.take(1)
        test_data = dataset.take(BATCH_SIZE).repeat(3).batch(BATCH_SIZE, drop_remainder=True)

        model = self.build_model(self.batch_size)

        ini_epoch = 0
        if cont:
            latest = self.get_checkpoint(checkpoint_num)
            if latest is not None:
                ini_epoch = int(latest[self.check_dir+6:])
                model.load_weights(latest)
        else:
            AiInterface.remove_checkpoints(self.check_dir)

        model.compile(optimizer="adam",
                      loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True, name="loss"),
                      metrics="accuracy",)
        checkpoint_call_back = tf.keras.callbacks.ModelCheckpoint(self.check_dir+"/ckpt_{epoch}", save_weights_only=True)

        model.fit(train_data,
                  epochs=epochs + ini_epoch, initial_epoch=ini_epoch,
                  callbacks=[checkpoint_call_back,
                             self.loss_callback(),
                             self.get_scheduler(lr, cont=cont),],
                  verbose=2,
                  validation_data=test_data,
                  validation_freq=3,
                  )

        return model

    def predict_vel(self, matrix, temperature, model=None, checkpoint_num=None):
        checkpoint = self.get_checkpoint(checkpoint_num)
        if model is None:
            model = self.build_model(1)
            model.load_weights(checkpoint).expect_partial()

        input_eval = np.array(matrix, dtype=np.float32)  # Formatting the start string
        input_eval = tf.expand_dims(input_eval, 0)

        model.reset_states()
        prediction = model(input_eval)
        prediction = tf.squeeze(prediction, 0) / temperature
        prediction = tf.random.categorical(prediction, num_samples=1).numpy()[-1, 0]

        return prediction

    def parse_start(self, start):
        raise Exception("This model does not support this operation")

    def generate_text(self, model, num, start, temperature) -> list:
        raise Exception("This model does not support this operation")

    def guess(self, num=10000, start=None, temp=0.8, model=None, checkpoint_num=None) -> list:
        raise Exception("This model does not suppoprt this operation")

if __name__ == "__main__":
    ai = Ai()
    # ai.process_all()
    # converted = ai.midi_to_data(mido.MidiFile("midis/alb_esp1.mid"))
    dataset = ai.get_dataset()
    for d in dataset.take(1):
        print(d)

    # ai.train(1, cont=False)

