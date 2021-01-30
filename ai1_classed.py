import os
from typing import Union

import mido

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
import numpy as np
import shutil
from math import isclose
import pickle
import ast

from AiInterface import AiInterface

try:
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
except:
    print("No GPU")

TICKS_PER_BEAT = 512
BATCH_SIZE = 256
SEQ_LENGTH = 200


class Ai1(AiInterface):
    def __init__(self):
        super().__init__("checkpoints", "data", "ai1_vocab.pkl", BATCH_SIZE, TICKS_PER_BEAT)

    def midi_to_data(self, midi: mido.MidiFile, vocabs: list) -> (np.array, list):
        if not vocabs:
            vocabs = [[-1], [-1], [-1], [-1, 0]]
        ticks_per_beat = midi.ticks_per_beat
        simple = [[-1, -1, -1, -1]]
        offset = 0

        for i, msg in enumerate(mido.merge_tracks(midi.tracks)):
            if msg.type[:4] == "note":
                note = msg.note
                vel = msg.velocity
                time = msg.time / ticks_per_beat + offset

                if vel != 0:
                    if note not in vocabs[0]:
                        vocabs[0].append(note)
                    if vel not in vocabs[1]:
                        vocabs[1].append(vel)
                    time = round(time, 6)
                    if time not in vocabs[2]:
                        vocabs[2].append(time)
                    simple.append([note, vel, time, 0])
                    offset = 0

                else:
                    offset = time
                    ind = len(simple) - 1
                    length = time
                    # Loop through end of list until all note with current node value's length is set
                    change_ind = None
                    last_length = None

                    while ind >= 0:
                        if simple[ind][0] == note:
                            if simple[ind][3] == 0:
                                change_ind = ind
                                last_length = length
                            elif change_ind is not None:
                                length = last_length
                                length = round(length, 5)
                                if length not in vocabs[3]:
                                    vocabs[3].append(length)
                                simple[change_ind][3] = length
                                break

                        elif ind == 0 and change_ind is not None:
                            length = last_length
                            length = round(length, 5)
                            if length not in vocabs[3]:
                                vocabs[3].append(length)
                            simple[change_ind][3] = length
                            break

                        time = simple[ind][2]
                        length += time
                        ind -= 1

        for i, event in enumerate(simple):
            simple[i] = [vocabs[j].index(x) for j, x in enumerate(event)]

        return np.array(simple), vocabs

    def data_to_midi_sequence(self, sequence: list) -> list:
        vocabs = self.vocabs
        for i, data in enumerate(sequence):
            sequence[i] = [vocabs[j][x] for j, x in enumerate(data)]

        sequence = [x for x in sequence if -1 not in x]
        i = 0
        while i < len(sequence):
            data = sequence[i]

            note, vel, time = data[0], data[1], data[2]
            data[2] = time * self.ticks_per_beat

            if vel != 0:
                length = data[3]
                insert_ind = i + 1
                while insert_ind < len(sequence) and length > sequence[insert_ind][2]:
                    length -= sequence[insert_ind][2]
                    insert_ind += 1
                if not (insert_ind < len(sequence) and sequence[insert_ind][0] == note and sequence[insert_ind][
                    1] == 0):
                    if insert_ind < len(sequence):
                        sequence[insert_ind][2] -= length
                    sequence.insert(insert_ind, [note, 0, length])
                del data[3]

            i += 1

        for i, item in enumerate(sequence):
            sequence[i] = [round(x) for x in item]

        return sequence

    def get_dataset(self) -> Union[np.array, tf.data.Dataset]:
        data = []

        for file in os.listdir(self.data_dir):
            file_name = os.fsdecode(file)
            with(open(os.path.join(self.data_dir, file_name), "rb")) as f:
                data.append(np.load(f)['arr_0'])

        data = np.vstack(data)
        return data

    def build_model(self, batch_size) -> keras.Model:
        notes, vels, times, lengths = self.vocabs

        inputs = keras.layers.Input(batch_shape=(batch_size, None, 4), batch_size=batch_size)
        note_x, vel_x, time_x, length_x = tf.split(inputs, 4, -1)

        emb_dim1 = 32
        note_x1 = keras.layers.Embedding(128, emb_dim1)(note_x)
        note_x1 = keras.layers.Reshape((-1, emb_dim1))(note_x1)
        note_x1 = keras.layers.Dropout(0.2)(note_x1)

        emb_dim2 = 32
        vel_x1 = keras.layers.Embedding(128, emb_dim2)(vel_x)
        vel_x1 = keras.layers.Reshape((-1, emb_dim2))(vel_x1)
        vel_x1 = keras.layers.Dropout(0.2)(vel_x1)

        emb_dim3 = 64
        time_x1 = keras.layers.Embedding(1000, emb_dim3)(time_x)
        time_x1 = keras.layers.Reshape((-1, emb_dim3))(time_x1)
        time_x1 = keras.layers.Dropout(0.2)(time_x1)

        emb_dim4 = 128
        length_x1 = keras.layers.Embedding(2500, emb_dim4)(length_x)
        length_x1 = keras.layers.Reshape((-1, emb_dim4))(length_x1)
        length_x1 = keras.layers.Dropout(0.2)(length_x1)

        note_x1 = keras.layers.GRU(128, return_sequences=True, stateful=True, dropout=0.2, kernel_regularizer="l2")(
            note_x1)
        vel_x1 = keras.layers.GRU(128, return_sequences=True, stateful=True, dropout=0.2, kernel_regularizer="l2")(
            vel_x1)
        time_x1 = keras.layers.GRU(128, return_sequences=True, stateful=True, dropout=0.2, kernel_regularizer="l2")(
            time_x1)
        length_x1 = keras.layers.GRU(128, return_sequences=True, stateful=True, dropout=0.2, kernel_regularizer="l2")(
            length_x1)

        y1 = keras.layers.concatenate((note_x1, vel_x1, time_x1, length_x1), -1)
        y1 = keras.layers.Dropout(0.3)(y1)

        y1 = keras.layers.GRU(512, return_sequences=True, stateful=True, dropout=0.2, kernel_regularizer="l2")(y1)
        y1 = keras.layers.BatchNormalization()(y1)


        y_1 = keras.layers.Dense(len(notes), name="note")(y1)
        y_2 = keras.layers.Dense(len(vels), name="vel")(y1)
        y_3 = keras.layers.Dense(len(times), name="time")(y1)  # (keras.layers.Dropout(0.3)(y1))
        y_4 = keras.layers.Dense(len(lengths), name="length")(y1)  # (keras.layers.Dropout(0.3)(y1))

        m = keras.Model(inputs=inputs, outputs=[y_1, y_2, y_3, y_4])
        return m

    def train(self, epochs=10, cont=False, lr=0.001, checkpoint_num=None):
        data = self.get_dataset()
        X, temp_y = data[:-1], data[1:]
        note_y = temp_y[:, 0]
        vel_y = temp_y[:, 1]
        time_y = temp_y[:, 2]
        length_y = temp_y[:, 3]
        dataset = tf.data.Dataset.from_tensor_slices(
            (X, {"note": note_y, "vel": vel_y, "time": time_y, "length": length_y}))
        dataset = dataset.batch(SEQ_LENGTH, drop_remainder=True)
        train_data = dataset.skip(500) \
            .batch(50).shuffle(10000, reshuffle_each_iteration=True).unbatch() \
            .repeat().batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
        test_data = dataset.take(500).batch(BATCH_SIZE, drop_remainder=True)

        model = self.build_model(self.batch_size)

        ini_epoch = 0
        if cont:
            latest = self.get_checkpoint(checkpoint_num)
            if latest is not None:
                ini_epoch = int(latest[17:])
                model.load_weights(latest)
        else:
            AiInterface.remove_checkpoints(self.check_dir)

        model.compile(optimizer="adam",
                      loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True, name="loss"),
                      metrics="accuracy", )
        checkpoint_call_back = tf.keras.callbacks.ModelCheckpoint(self.check_dir + "/ckpt_{epoch}",
                                                                  save_weights_only=True)

        model.fit(train_data,
                  epochs=epochs + ini_epoch, initial_epoch=ini_epoch,
                  callbacks=[self.get_scheduler(lr, cont=cont),
                             checkpoint_call_back,
                             self.loss_callback()],
                  verbose=2,
                  validation_data=test_data,
                  validation_freq=3,
                  steps_per_epoch=int(20000 / BATCH_SIZE))
        return model

    def generate_text(self, model, num, start, temperature) -> list:
        vocabs = self.vocabs
        for i, item in enumerate(start):
            start[i] = [vocabs[j].index(x) for j, x in enumerate(item)]
        input_eval = np.array([start])  # Formatting the start string
        # input_eval = tf.expand_dims(input_eval, 0)

        text_generated = []

        model.reset_states()
        for i in range(num):
            predictions = model(input_eval)
            note_predict, vel_predict, time_predict, length_predict = predictions[0], predictions[1], predictions[
                2], \
                                                                      predictions[3]
            note_predict, vel_predict, time_predict, length_predict = tf.squeeze(note_predict, 0), tf.squeeze(
                vel_predict,
                0), \
                                                                      tf.squeeze(time_predict, 0), tf.squeeze(
                length_predict, 0)
            note_predict, vel_predict, time_predict, length_predict = note_predict / temperature, vel_predict / temperature, time_predict / temperature, length_predict / temperature
            note_id, vel_id, time_predict, length_predict = \
                tf.random.categorical(note_predict, num_samples=1).numpy()[
                    -1, 0], \
                tf.random.categorical(vel_predict, num_samples=1).numpy()[
                    -1, 0], \
                tf.random.categorical(time_predict, num_samples=1).numpy()[
                    -1, 0], \
                tf.random.categorical(length_predict, num_samples=1).numpy()[
                    -1, 0]

            input_eval = tf.expand_dims(np.array([note_id, vel_id, time_predict, length_predict]), 0)

            add = [note_id, vel_id, time_predict, length_predict]
            if -1 not in add:
                text_generated.append(add)
            else:
                num += 1

        return start + text_generated

    def guess(self, num=10000, start=None, temp=0.8, model=None, checkpoint_num=None) -> list:
        checkpoint = self.get_checkpoint(checkpoint_num)
        if model is None:
            model = self.build_model(1)
            model.load_weights(checkpoint).expect_partial()

        if start is None:
            start = [[76, 60, 0, 1], [74, 60, 0, 1]]
        model.build(tf.TensorShape([1, None, 4]))
        generated = self.generate_text(model, num, start, temp)
        return generated


if __name__ == "__main__":
    ai = Ai1()
    # ai.train(1, cont=False)
    # ai.process_all()
    converted = ai.midi_to_data(mido.MidiFile("midis/alb_esp1.mid"), ai.vocabs)
    unconverted = ai.data_to_midi_sequence(list(converted[0]))
    ai.make_midi_file(unconverted, "temp.mid")
    # notes = ai.guess(100)
    # notes = ai.data_to_midi_sequence(notes)
    # ai.make_midi_file(notes, "temp.mid")