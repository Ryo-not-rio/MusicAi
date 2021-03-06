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
from basicAiInterface import Interface

try:
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
except:
    print("No GPU")

BATCH_SIZE = 128
SEQ_LENGTH = 200

class Ai(Interface):
    def __init__(self):
        super().__init__("vel_checkpoints", "vel_data3", BATCH_SIZE)
        self.loss = -1
        self.prev_loss = 100
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

    def midi_to_data(self, midi: mido.MidiFile) -> (np.array, list):
        ticks_per_beat = midi.ticks_per_beat
        simple_seq = [[-1, -1, -1]]
        vels = [0]
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
        for i, event in enumerate(simple_seq):
            note, time, length = event
            next_matrix = matrix_seq[-1][:]
            next_matrix = [x-time if x-time > 0 else 0 for x in next_matrix]
            next_matrix[note] = max(next_matrix[note], length)
            if i > 0:
                matrix_seq.append(next_matrix)

        return [np.array(matrix_seq), np.array(vels)]

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

        y = keras.layers.Dropout(0.1)(y)
        y = keras.layers.GRU(1024, stateful=True, return_sequences=True)(y)
        y = keras.layers.GRU(1024, stateful=True, return_sequences=True)(y)
        y = keras.layers.GRU(1024, stateful=True, return_sequences=True)(y)
        y = keras.layers.Dropout(0.3)(y)

        y_1 = keras.layers.Dense(128)(y)

        m = keras.Model(inputs=inputs, outputs=y_1)
        # m.summary()
        return m

    def train(self, epochs=10, cont=False, lr=0.001, checkpoint_num=None):
        dataset = self.get_dataset()
        dataset = dataset.batch(SEQ_LENGTH, drop_remainder=True)
        train_data = dataset.skip(BATCH_SIZE).batch(BATCH_SIZE, drop_remainder=True)
        test_data = dataset.take(BATCH_SIZE).repeat(3).batch(BATCH_SIZE, drop_remainder=True)

        model = self.build_model(self.batch_size)

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
                  verbose=2,
                  validation_data=test_data,
                  validation_freq=3,
                  )

        return model

    @staticmethod
    def predict_vel(matrix, temperature, model):
        input_eval = np.array(matrix, dtype=np.float32)  # Formatting the start string
        input_eval = tf.expand_dims([input_eval], 0)

        prediction = model(input_eval)
        input_eval = tf.squeeze(prediction)
        prediction = input_eval / temperature
        prediction = tf.random.categorical([prediction], num_samples=1).numpy()[-1, 0]

        return prediction, model

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

    ai.train(100, cont=False)

