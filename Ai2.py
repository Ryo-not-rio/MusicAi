import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
import numpy as np
import random
import threading
import queue
import mido
import shutil
import time
import pickle
import random

from AiInterface import AiInterface

try:
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
except:
    print("No GPU")

BATCH_SIZE = 128
TICKS_PER_BEAT = 128
SEQ_LENGTH = TICKS_PER_BEAT * 4 * 2


def split_input(chunk):
    return chunk[:-1], chunk[1:]


class Ai(AiInterface):
    def __init__(self):
        super().__init__("./checkpoints2", "data2", "ai2_vocab.pkl", BATCH_SIZE, TICKS_PER_BEAT)

    def midi_to_data(self, mid, vocabs) -> (np.array, list):
        if not vocabs:
            vocabs = [["-1"]]
        vocab = vocabs[0]
        sequence = [[vocab.index("-1")] * 128]
        ticks_per_beat = mid.ticks_per_beat
        mult = self.ticks_per_beat / ticks_per_beat

        start = False
        step_matrix = ["0"] * 128
        playing = [0] * 128
        prev_note = None
        for i, msg in enumerate(mido.merge_tracks(mid.tracks)):
            if msg.type[:4] == "note":
                note = msg.note
                vel = msg.velocity
                time = round(msg.time * mult)

                if time > 0 and start:
                    add = []
                    for n, xs in enumerate(step_matrix[:]):
                        if xs not in vocab:
                            vocab.append(xs)
                        add.append(vocab.index(xs))
                    sequence.append(add[:])

                    for j, value in enumerate(add):
                        if vocab[value][-1] == "2":
                            if "1" not in vocab: vocab.append("1")
                            add[j] = vocab.index("1")
                        else:
                            add[j] = value
                    sequence += [add[:]] * (time - 1)

                if vel != 0:
                    if time == 0 and prev_note == note:
                        step_matrix[note] += "2"
                    else:
                        step_matrix[note] = "2"
                    playing[note] += 1
                else:
                    if playing[note] > 0:
                        playing[note] -= 1

                    if time == 0 and prev_note == note:
                        step_matrix[note] += "3"
                    else:
                        step_matrix[note] = "3"

                if not start: start = True
                prev_note = note

        sequence.append([vocab.index(x) for x in step_matrix[:]])

        return np.array(sequence), vocabs

    def data_to_midi_sequence(self, sequence):
        idx2note = np.array(self.vocabs[0])
        vec_decode = np.vectorize(lambda x: idx2note[int(x)])
        sequence = vec_decode(np.array(sequence)).tolist()
        notes = []
        playing = [0] * 128
        time = 0
        for matrix in sequence:
            for i, value in enumerate(matrix):
                for v in value:
                    if v == "2":
                        notes.append([i, 80, time])
                        playing[i] += 1
                        time = 0

                    elif v == "3":
                        # if playing[i] > 0:
                        playing[i] -= 1
                        notes.append([i, 0, time])
                        time = 0

            time += 1

        for note, value in enumerate(playing):
            if value > 0:
                notes.append([note, 0, 0])
        return notes

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
                np.savez_compressed(f, data)

        with open(self.vocab_file, "wb") as f:
            pickle.dump(vocabs, f)

        self.vocabs = vocabs
        print("processed all midi files.")
        print("time_taken: ", time.time() - start)
        return vocabs

    def get_dataset(self):
        def load_file(file):
            f = np.load(os.path.join(self.data_dir, file))
            arr = f['arr_0']
            f.close()
            return arr

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

                if thread_count < 5:
                    t = threading.Thread(target=lambda q, f: q.put(load_file(f)),
                                         args=(que, file_name))
                    t.start()
                    thread_count += 1
                try:
                    arr = que.get(block=False)
                    thread_count -= 1
                    yield arr[:-1], arr[1:]
                except queue.Empty:
                    pass

                prev_file = file

        return tf.data.Dataset.from_generator(load_raw_data,
                                              output_types=(tf.uint8, tf.uint8),
                                              output_shapes=(
                                              tf.TensorShape([None, 128]), tf.TensorShape([None, 128]))).unbatch()

    def build_model(self, batch_size):
        inputs = keras.layers.Input(batch_shape=(batch_size, None, 128), batch_size=batch_size)

        y = keras.layers.GRU(1024, batch_size=batch_size, return_sequences=True, stateful=True, dropout=0.2,
                             kernel_regularizer="l2")(inputs)
        y = keras.layers.Dropout(0.3)(y)
        y = keras.layers.Dense(128 * len(self.vocabs[0]))(y)
        y = keras.layers.Reshape((-1, 128, len(self.vocabs[0])))(y)

        m = keras.Model(inputs=inputs, outputs=y)
        # m.summary()
        return m

    def train(self, epochs=10, cont=False, lr=0.001, checkpoint_num=None):
        data = self.get_dataset()
        data = data.batch(SEQ_LENGTH, drop_remainder=True)
        train_data = data.skip(2000).repeat()
        train_data = train_data.batch(BATCH_SIZE, drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE)
        test_data = data.take(2000).batch(BATCH_SIZE, drop_remainder=True).prefetch(1)

        model = self.build_model(BATCH_SIZE)

        ini_epoch = 0
        if cont:
            latest = self.get_checkpoint(checkpoint_num)
            if latest is not None:
                ini_epoch = int(latest[20:])
                model.load_weights(latest)
        else:
            AiInterface.remove_checkpoints(self.check_dir)

        model.compile(optimizer="adam",
                      loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics="accuracy", )
        checkpoint_call_back = tf.keras.callbacks.ModelCheckpoint("checkpoints2/ckpt_{epoch}", save_weights_only=True)

        model.fit(train_data,
                  epochs=epochs + ini_epoch, initial_epoch=ini_epoch,
                  callbacks=[self.get_scheduler(lr, cont), checkpoint_call_back, self.loss_callback()],
                  verbose=1,
                  validation_data=test_data,
                  validation_freq=3,
                  steps_per_epoch=int(40000 / BATCH_SIZE))
        return model

    def parse_start(self, start_seq: list) -> np.array:
        vocab = self.vocabs[0]
        sequence = [[vocab.index("-1")] * 128]

        start = False
        step_matrix = ["0"] * 128
        playing = [0] * 128
        prev_note = None
        for i, v in enumerate(start_seq):
            note, vel, time = v

            if time > 0 and start:
                add = []
                for n, xs in enumerate(step_matrix[:]):
                    if xs not in vocab:
                        vocab.append(xs)
                    add.append(vocab.index(xs))
                sequence.append(add[:])

                for j, value in enumerate(add):
                    if vocab[value][-1] == "2":
                        add[j] = vocab.index("1")
                    else:
                        add[j] = value
                sequence += [add[:]] * (time - 1)

            if vel != 0:
                if time == 0 and prev_note == note:
                    step_matrix[note] += "2"
                else:
                    step_matrix[note] = "2"
                playing[note] += 1
            else:
                if playing[note] > 0:
                    playing[note] -= 1

                if time == 0 and prev_note == note:
                    step_matrix[note] += "3"
                else:
                    step_matrix[note] = "3"

            if not start: start = True
            prev_note = note

        return np.array(sequence), playing

    def generate_text(self, model, num, start, temperature):
        print("generating...")
        input_eval, playing = self.parse_start(start)

        text_generated = list(input_eval[1:])

        input_eval = tf.expand_dims(input_eval, 0)
        model.reset_states()

        vocab = self.vocabs[0]
        note2idx = {k: i for i, k in enumerate(vocab)}
        counts2 = [x.count("2") for x in vocab]
        counts3 = [x.count("3") for x in vocab]

        def parse_func(playing_note, v):
            v = int(v)
            value = vocab[v]
            playing_note += counts2[v]
            playing_note -= counts3[v]
            if playing_note > 0 and value[-1] == "0":
                new_value = value[:-1] + "1"
                while new_value not in note2idx:
                    if new_value[0] == "2":
                        playing_note -= 1
                    elif new_value[0] == "3":
                        playing_note += 1
                    new_value = new_value[1:]
                v = note2idx[new_value]
            elif playing_note <= 0 and value[-1] == "1":
                new_value = value[:-1] + "0"
                while new_value not in note2idx:
                    if new_value[0] == "2":
                        playing_note -= 1
                    elif new_value[0] == "3":
                        playing_note += 1
                    new_value = new_value[1:]
                v = note2idx[new_value]

            if playing_note < 0: playing_note = 0
            return playing_note, v

        vec_parse_func = np.vectorize(parse_func)

        for i in range(num):
            if i % 150 == 0:
                print(f"Generating {i}/{num}")
            predictions = model.predict(input_eval)
            predictions = tf.squeeze(predictions, 0)
            predictions = predictions[-1]

            predictions = tf.map_fn(
                lambda x: tf.squeeze(
                    tf.random.categorical(
                        tf.expand_dims(x / temperature, 0
                                       ), num_samples=1, dtype=tf.int32
                    )
                ),
                predictions, fn_output_signature=tf.int32)

            playing, predictions = vec_parse_func(playing, predictions)
            input_eval = tf.expand_dims([predictions], 0)

            add = predictions.tolist()
            text_generated.append(add)

        return text_generated

    def guess(self, num=10000, start=None, temp=0.8, model=None, checkpoint_num=None):
        if model is None:
            model = self.build_model(1)
            model.load_weights(self.get_checkpoint(checkpoint_num)).expect_partial()

        if start is None:
            start = [[76, 60, 0], [76, 0, 256]]

        model.build(tf.TensorShape([1, None, 128]))
        generated = self.generate_text(model, num, start, temp)
        return generated


if __name__ == "__main__":
    ai = Ai()
    ai.process_all()

    # converted, vocabs = ai.midi_to_data(mido.MidiFile("./midis/alb_esp1.mid"), [])
    # ai.vocabs = vocabs
    # # print(converted.shape)
    # noted = ai.data_to_midi_sequence(converted.tolist())
    # # print(noted)
    # ai.make_midi_file(noted, "temp.mid")

    # ai.train(1)

    notes = ai.guess(100)
    notes = ai.data_to_midi_sequence(notes)
    ai.make_midi_file(notes, "temp.mid")
