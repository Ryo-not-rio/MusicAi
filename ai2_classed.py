import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
import numpy as np
import shutil
from math import log
import pickle
import random
import threading
import queue
import mido

import preprocess
from AiInterface import AiInterface


try:
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
except:
    print("No GPU")

SEQ_LENGTH = 100
BATCH_SIZE = 64
BASE_TICKS_PER_BEAT = 256

def split_input(chunk):
    return chunk[:-1], chunk[1:]

class MatrixAi(AiInterface):
    def __init__(self):
        super().__init__("./checkpoints2", "data2", "matrix_vocab.pkl", BATCH_SIZE)
        self.vocab = self.vocabs[0]
        self.note2idx = {i: k for k, i in enumerate(self.vocab)}
        self.idx2note = np.array(self.vocab)

    def midi_to_data(self, mid, vocabs) -> (np.array, list):
        if not vocabs:
            vocabs = [[-1]]
        vocab = vocabs[0]
        sequence = [[vocab.index(-1)]*127]
        ticks_per_beat = mid.ticks_per_beat
        mult = BASE_TICKS_PER_BEAT / ticks_per_beat

        start = False
        step_matrix = [0] * 127
        for i, msg in enumerate(mido.merge_tracks(mid.tracks)):
            if msg.type[:4] == "note":
                note = msg.note
                vel = msg.velocity
                time = round(msg.time * mult)

                if time > 0 and start:
                    add = []
                    for x in step_matrix[:]:
                        if x in vocab:
                            add.append(vocab.index(x))
                        else:
                            vocab.append(x)
                            add.append(len(vocab)-1)
                    sequence.append(add)
                    step_matrix = [0 if x == 0 else 1 for x in step_matrix]
                    sequence += [step_matrix[:]] * (time - 1)

                if vel != 0:
                    step_matrix[note] = 2
                else:
                    step_matrix[note] = 0

                if not start: start = True

        sequence.append(step_matrix)

        return np.array(sequence), vocabs

    def data_to_midi_sequence(self, sequence):
        notes = []
        playing = [0] * 127
        time = 0
        for matrix in sequence:
            for i, v in enumerate(matrix):
                count = 0
                if v == 2:
                    notes.append([i, 80, time])
                    playing[i] += 1
                    time = 0
                    count += 1

                elif v == 0 and playing[i]:
                    notes.append([i, 0, time])
                    playing[i] -= 1
                    time = 0
                    count += 1

            time += 1

        return notes

    def get_dataset(self):
        def load_file(file):
            with(open(os.path.join(self.data_dir, file), "rb")) as f:
                arr = np.load(f, allow_pickle=True)
                return arr

        def load_raw_data():
            files_list = os.listdir(self.data_dir)
            random.shuffle(files_list)
            index = 0
            que = queue.Queue()
            thread_count = 0
            while True:
                file = files_list[index]
                file_name = os.fsdecode(file)

                if thread_count < 5:
                    t = threading.Thread(target=lambda q, f: q.put(load_file(f)),
                                         args=(que, file_name))
                    t.start()
                    thread_count += 1
                    index += 1

                try:
                    arr = que.get(block=False)
                    thread_count -= 1
                    yield arr[:-1], arr[1:]
                except queue.Empty:
                    pass

                if index == len(files_list):
                    index = 0

        return tf.data.Dataset.from_generator(load_raw_data,
                                              output_types=(tf.uint8, tf.uint8),
                                              output_shapes=(tf.TensorShape([None, 127]), tf.TensorShape([None, 127]))).unbatch()

    def build_model(self, batch_size):
        inputs = keras.layers.Input(batch_shape=(batch_size, None, 127), batch_size=batch_size)

        y = keras.layers.GRU(512, batch_size=batch_size, return_sequences=True, stateful=True, dropout=0.2)(inputs)
        y = keras.layers.Dropout(0.3)(y)
        y = keras.layers.Dense(127 * len(self.vocab))(y)
        y = keras.layers.Reshape((-1, 127, len(self.vocab)))(y)

        m = keras.Model(inputs=inputs, outputs=y)
        m.summary()
        return m

    def train(self, epochs=10, cont=False, lr=0.001, checkpoint_num=None):
        data = self.get_dataset()
        data = data.batch(SEQ_LENGTH, drop_remainder=True)
        train_data = data.skip(10 * BATCH_SIZE).repeat()
        # train_data = train_data.batch(150*midi_stuff.BASE_TICKS_PER_BEAT).shuffle(10, reshuffle_each_iteration=True).unbatch()
        train_data = train_data.batch(BATCH_SIZE, drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE)
        test_data = data.take(10 * BATCH_SIZE).batch(BATCH_SIZE, drop_remainder=True)

        model = self.build_model(BATCH_SIZE)

        ini_epoch = 0
        if cont:
            latest = self.get_checkpoint(checkpoint_num)
            if latest is not None:
                ini_epoch = int(latest[17:])
                model.load_weights(latest)
        else:
            AiInterface.remove_checkpoints(self.check_dir)

        model.compile(optimizer="adam",
                      loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics="accuracy", )
        checkpoint_call_back = tf.keras.callbacks.ModelCheckpoint("checkpoints2/ckpt_{epoch}", save_weights_only=True,
                                                                  save_freq=3)

        model.fit(train_data,
                  epochs=epochs + ini_epoch, initial_epoch=ini_epoch,
                  callbacks=[self.get_scheduler(lr, cont), checkpoint_call_back, self.loss_callback()],
                  verbose=1,
                  validation_data=test_data,
                  validation_freq=3,
                  steps_per_epoch=10)
        return model

    def parse_start(self, start_seq: list) -> np.array:
        sequence = [[-1] * 127]

        start = False
        step_matrix = [0] * 127
        for i, v in enumerate(start_seq):
            note, vel, time = v

            if time > 0 and start:
                sequence.append([self.note2idx[x] for x in step_matrix[:]])
                step_matrix = [0 if x == 0 else 1 for x in step_matrix]
                sequence += [step_matrix[:]] * (time - 1)

            if vel != 0:
                step_matrix[note] = 2
            else:
                step_matrix[note] = 0

            if not start: start = True

        return np.array(sequence)

    def generate_text(self, model, num, start, temperature):
        vec_decode = np.vectorize(lambda x: self.idx2note[int(x)])

        input_eval = self.parse_start(start)

        text_generated = list(vec_decode(input_eval[1:]))

        input_eval = tf.expand_dims(input_eval, 0)
        model.reset_states()

        for i in range(num):
            predictions = model.predict(input_eval)
            predictions = tf.squeeze(predictions, 0)
            predictions = predictions[-1]
            predictions = tf.map_fn(
                          lambda x: tf.random.categorical(tf.expand_dims(x, 0) / temperature, num_samples=1).numpy()[-1, 0],
                          predictions).numpy()

            print(predictions.shape)
            input_eval = tf.expand_dims([predictions], 0)

            add = vec_decode(predictions).tolist()
            text_generated.append(add)

        return text_generated

    def guess(self, num=10000, start=None, temp=0.8, model=None, checkpoint_num=None):
        if model is None:
            model = self.build_model(1)
            model.load_weights(self.get_checkpoint(checkpoint_num)).expect_partial()

        if start is None:
            start = [[76, 60, 0], [76, 0, 256]]

        model.build(tf.TensorShape([1, None, 127]))
        generated = self.generate_text(model, num, start, temp)
        return generated


if __name__ == "__main__":
    ai = MatrixAi()
    # ai.process_all()
    # ai.train(1, cont=False)
    generated = ai.guess(num=10)
    print(np.array(generated))

