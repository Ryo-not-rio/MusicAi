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
import queue
import threading

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

class Ai3Vels(AiInterface):
    def __init__(self):
        super().__init__("checkpoints3_vels", "data3_vels", "ai3_vels_vocab.pkl", BATCH_SIZE, TICKS_PER_BEAT)

    def get_matrix(self, seq):
        matrix_seq = [[-1] * 256]
        for i, event in enumerate(seq):
            note, vel, time, length = event
            next_matrix = matrix_seq[-1][:]
            for j, x in enumerate(next_matrix):
                if j % 2 == 0:
                    if x - time > 0:
                        next_matrix[j] = x - time
                    else:
                        next_matrix[j] = 0
                        next_matrix[j + 1] = 0

            next_matrix[note * 2] = max(next_matrix[note * 2], length)
            if next_matrix[note*2] > 0:
                next_matrix[note * 2 + 1] = vel/128
            matrix_seq.append(next_matrix)
        return matrix_seq

    def midi_to_data(self, midi: mido.MidiFile, vocabs: list) -> (np.array, list):
        if not vocabs:
            vocabs = [[-1], [-1], [-1], [-1, 0]]
        ticks_per_beat = midi.ticks_per_beat
        simple_seq = [[-1, -1, -1, -1]]
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
                    simple_seq.append([note, vel, time, 0])
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
                        if length not in vocabs[3]:
                            vocabs[3].append(length)
                        simple_seq[change_ind][3] = length

                    while ind >= 0:
                        if simple_seq[ind][0] == note:
                            if simple_seq[ind][3] == 0:
                                change_ind = ind
                                last_length = length
                            elif change_ind is not None:
                                update_length(last_length)
                                break

                        elif ind == 0 and change_ind is not None:
                            update_length(last_length)
                            break

                        time = simple_seq[ind][2]
                        length += time
                        ind -= 1

        matrix_seq = self.get_matrix(simple_seq[:-1])

        for i, event in enumerate(simple_seq):
            simple_seq[i] = [vocabs[j].index(x) for j, x in enumerate(event)]

        return [np.array(matrix_seq), np.array(simple_seq)], vocabs

    def data_to_midi_sequence(self, sequence: list) -> list:
        for i, d in enumerate(sequence):
            sequence[i] = [self.vocabs[j][x] for j, x in enumerate(d)]
        sequence = [x for x in sequence if -1 not in x]
        i = 0
        while i < len(sequence):
            data = sequence[i]

            note, vel, time = data[0], data[1], data[2]
            data[2] = time * self.ticks_per_beat

            if vel != 0:
                length = data[3]
                insert_ind = i+1
                while insert_ind < len(sequence) and length > sequence[insert_ind][2]:
                    length -= sequence[insert_ind][2]
                    insert_ind += 1
                if not (insert_ind < len(sequence) and sequence[insert_ind][0] == note and sequence[insert_ind][1] == 0):
                    if insert_ind < len(sequence):
                        sequence[insert_ind][2] -= length
                    sequence.insert(insert_ind, [note, 0, length])
                del data[3]

            i += 1

        for i, item in enumerate(sequence):
            sequence[i] = [round(x) for x in item]

        return sequence

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
        def load_file(file):
            f = np.load(os.path.join(self.data_dir, file))
            arr0, arr1 = f['arr_0'], f['arr_1']
            f.close()
            return arr0, arr1

        def load_raw_data():
            files_list = os.listdir(self.data_dir)
            que = queue.Queue()
            thread_count = 0
            prev_file = None
            while True:
                file = random.choice(files_list)
                while file == prev_file:
                    file = random.choice(files_list)
                file_name = os.fsdecode(file)

                if thread_count < 3:
                    t = threading.Thread(target=lambda q, f: q.put(load_file(f)),
                                         args=(que, file_name))
                    t.start()
                    thread_count += 1
                try:
                    X, y = que.get(block=False)
                    thread_count -= 1
                    yield X, {"notes": y[:, 0], "vels": y[:, 1], "times": y[:, 2], "lengths": y[:, 3]}
                except queue.Empty:
                    pass

                prev_file = file

        return tf.data.Dataset.from_generator(load_raw_data,
                                              output_types=(tf.float64, {"notes": tf.int32, "vels": tf.int32, "times": tf.int32, "lengths": tf.int32}),
                                              output_shapes=(tf.TensorShape((None, 256)),
                                                             {"notes": ((None, )), "vels": ((None, )), "times": ((None, )), "lengths": ((None, ))})
                                              ).unbatch()

    def build_model(self, batch_size) -> keras.Model:
        notes, vels, times, lengths = self.vocabs

        inputs = keras.layers.Input(batch_shape=(batch_size, None, 256), batch_size=batch_size)
        y = inputs
        y = keras.layers.Dense(1024, activation="tanh")(y)
        y = keras.layers.Dropout(0.2)(y)
        # embedding_dim = 1
        # y = keras.layers.Reshape((-1, 128*embedding_dim))(y)
        y = keras.layers.GRU(512, stateful=True, return_sequences=True)(y)
        y = keras.layers.GRU(1024, stateful=True, return_sequences=True)(y)
        # y = keras.layers.GRU(1024, stateful=True, return_sequences=True, return_state=False)(y)
        y = keras.layers.Dropout(0.3)(y)

        y_1 = keras.layers.Dense(len(notes), name="notes")(y)
        y_2 = keras.layers.Dense(len(vels), name="vels")(y)
        y_3 = keras.layers.Dense(len(times), name="times")(y)
        y_4 = keras.layers.Dense(len(lengths), name="lengths")(y)

        m = keras.Model(inputs=inputs, outputs=[y_1, y_2, y_3, y_4])
        # m.summary()
        return m

    def train(self, epochs=10, cont=False, lr=0.001, checkpoint_num=None):
        dataset = self.get_dataset()
        dataset = dataset.batch(SEQ_LENGTH, drop_remainder=True)
        train_data = dataset.skip(BATCH_SIZE).batch(BATCH_SIZE, drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE)
        test_data = dataset.take(BATCH_SIZE).repeat(3).batch(BATCH_SIZE, drop_remainder=True)

        model = self.build_model(self.batch_size)

        ini_epoch = 0
        if cont:
            latest = self.get_checkpoint(checkpoint_num)
            if latest is not None:
                ini_epoch = int(latest[18:])
                model.load_weights(latest)
        else:
            AiInterface.remove_checkpoints(self.check_dir)

        model.compile(optimizer="adam",
                      loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True, name="loss"),
                      metrics="accuracy",
                      loss_weights=[7, 14, 0.6, 4.5],)
        checkpoint_call_back = tf.keras.callbacks.ModelCheckpoint(self.check_dir+"/ckpt_{epoch}", save_weights_only=True)

        model.fit(train_data,
                  epochs=epochs + ini_epoch, initial_epoch=ini_epoch,
                  callbacks=[checkpoint_call_back,
                             self.loss_callback(),
                             self.get_scheduler(lr, cont=cont),],
                  verbose=2,
                  validation_data=test_data,
                  validation_freq=3,
                  steps_per_epoch=39,
                  )

        return model

    def generate_text(self, model, num, start, temperature) -> list:
        vocabs = self.vocabs
        input_eval = self.get_matrix(start)
        input_eval = np.array(input_eval, dtype=np.float32)  # Formatting the start string
        input_eval = tf.expand_dims(input_eval, 0)

        text_generated = []

        model.reset_states()
        for i in range(num):
            predictions = model(input_eval)
            note_predict, vel_predict, time_predict, length_predict = predictions
            note_predict, vel_predict, time_predict, length_predict = tf.squeeze(note_predict, 0), \
                                                                      tf.squeeze(vel_predict, 0), \
                                                                      tf.squeeze(time_predict,0),\
                                                                      tf.squeeze(length_predict, 0)
            note_predict, vel_predict, time_predict, length_predict = note_predict / temperature, vel_predict/temperature, \
                                                                      time_predict / temperature, length_predict / temperature
            note_id, vel_id, time_id, length_id = tf.random.categorical(note_predict, num_samples=1).numpy()[-1, 0], \
                                                  tf.random.categorical(vel_predict, num_samples=1).numpy()[-1, 0], \
                                                  tf.random.categorical(time_predict, num_samples=1).numpy()[-1, 0], \
                                                  tf.random.categorical(length_predict, num_samples=1).numpy()[-1, 0]

            note, vel, time, length = vocabs[0][note_id], vocabs[1][vel_id], vocabs[2][time_id], vocabs[3][length_id]
            input_eval = input_eval[0][-1].numpy()

            for j, x in enumerate(input_eval):
                if j % 2 == 0:
                    if x - time > 0:
                        input_eval[j] = x - time
                    else:
                        input_eval[j] = 0
                        input_eval[j + 1] = 0

            input_eval[note * 2] = max(input_eval[note * 2], length)
            input_eval[note * 2 + 1] = self.vocabs[1].index(vel)

            input_eval = np.array([input_eval])
            input_eval = tf.expand_dims(input_eval, 0)

            add = [note_id, vel_id, time_id, length_id]
            if -1 not in add:
                text_generated.append(add)
            else:
                num += 1

        for i, event in enumerate(start):
            start[i] = [self.vocabs[j].index(x) for j, x in enumerate(event)]
        return start + text_generated

    def guess(self, num=10000, start=None, temp=0.8, model=None, checkpoint_num=None) -> list:
        checkpoint = self.get_checkpoint(checkpoint_num)
        if model is None:
            model = self.build_model(1)
            model.load_weights(checkpoint).expect_partial()

        if start is None:
            start = [[76, 70, 0, 1], [72, 70, 0, 1]]
        # model.build(tf.TensorShape([1, None, 128]))
        generated = self.generate_text(model, num, start, temp)
        return generated

if __name__ == "__main__":
    ai = Ai3Vels()
    # ai.process_all()
    print(ai.vocabs[1])
    # for d in ai.get_dataset().take(1):
    #     print(d[1]['vels'])
    # converted = ai.midi_to_data(mido.MidiFile("midis/alb_esp1.mid"), ai.vocabs)[0]
    # notes = ai.data_to_midi_sequence(list(converted[1]))
    # ai.train(1, cont=False)
    notes = ai.guess(100)
    notes = ai.data_to_midi_sequence(notes)
    print(notes)
    ai.make_midi_file(notes, "temp.mid")