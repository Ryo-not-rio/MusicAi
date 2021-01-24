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
import preprocess

try:
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
except:
    print("No GPU")


vectorized_func = np.vectorize(lambda x: note2idx[x])
def to_index(arr):
    return vectorized_func(arr)


vec_decode = np.vectorize(lambda x: idx2note[int(x)])
def decode(arr):
    return vec_decode(arr)


def split_input(chunk):
    return chunk[:-1], chunk[1:]

def load_raw_data():
    data = []
    directory = os.fsencode("data")

    for file in os.listdir(directory):
        file_name = os.fsdecode(file)
        if file_name[-3:] == "npy":
            with(open(os.path.join("data", file_name), "rb")) as f:
                data.append(np.load(f))
    return data

def pad(d):
    pad_length = seq_length*BATCH_SIZE - (d.shape[0] % (seq_length*BATCH_SIZE))
    d = np.concatenate((d, np.tile([[0, 0, 0, 0]], (pad_length, 1))))
    return d

def padded_data():
    data = load_raw_data()
    data = np.array(list(map(pad, data)))
    data = np.vstack(data)
    return data

def straight_data():
    return np.vstack(load_raw_data())

seq_length = 10
BATCH_SIZE = 50
repeats = 3

data = padded_data()

notes = sorted(set(data[:, 0]))
vels = sorted(set(data[:, 1]))
times = sorted(set(data[:, 2]))
lengths = sorted(set(data[:, 3]))
note2idx = {i: k for k, i in enumerate(notes)}
idx2note = np.array(notes)
vel2idx = {i: k for k, i in enumerate(vels)}
idx2vel = np.array(vels)
time2idx = {i: k for k, i in enumerate(times)}
idx2time = np.array(times)
length2idx = {i: k for k, i in enumerate(lengths)}
idx2length = np.array(lengths)

indexed = np.array(list(map(lambda x: to_index(x), data)))

X, temp_y = indexed[:-1], indexed[1:]
note_y = temp_y[:, 0]
vel_y = temp_y[:, 1]
time_y = temp_y[:, 2]
length_y = temp_y[:, 3]
data = tf.data.Dataset.from_tensor_slices((X, {"note": note_y, "vel": vel_y, "time": time_y, "length": length_y}))
# print(len(notes), len(vels), len(times), len(lengths))

data = data.repeat(repeats).batch(seq_length, drop_remainder=True)
train_data = data.skip(BATCH_SIZE*2).batch(BATCH_SIZE, drop_remainder=True)
test_data = data.take(BATCH_SIZE*2).batch(BATCH_SIZE, drop_remainder=True)

embed_dim = 30


def load_file(directory, file_name):
    directory = os.fsdecode(directory)
    if "vectorized" in file_name:
        with(open(os.path.join(directory, file_name), "rb")) as f:
            arr = np.load(f, allow_pickle=True)

            return arr
    try:
        with(open(os.path.join(directory, file_name[:-4]+"_vectorized.npy"), "rb")) as f:
            return np.load(f, allow_pickle=True)
    except FileNotFoundError:
        data = []
        data.append([-1] * 127)
        with(open(os.path.join(directory, file_name), "rb")) as f:
            data.append(np.load(f))

        arr = np.vstack(np.array(data))
        arr = to_index(arr)

        os.remove(os.path.join(directory, file_name))
        with(open(os.path.join(directory, file_name[:-4]+"_vectorized.npy"), "wb")) as f:
            np.save(f, arr)
        return arr


def load_raw_data():
    directory = os.fsencode("data2")
    files_list = os.listdir(directory)
    random.shuffle(files_list)
    index = 0
    que = queue.Queue()
    thread_count = 0
    while True:
        file = files_list[index]
        file_name = os.fsdecode(file)

        if thread_count < 5:
            t = threading.Thread(target=lambda q, d, f: q.put(load_file(d, f)), args=(que, directory, file_name))
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


def get_vocab(file_name):
    try:
        with open(file_name, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        preprocess.process()
        with open(file_name, "rb") as f:
            return pickle.load(f)


seq_length = 500
BATCH_SIZE = 512

vocab = get_vocab("matrix_vocab.pkl")
print(vocab)

note2idx = {i: k for k, i in enumerate(vocab)}
idx2note = np.array(vocab)

data = tf.data.Dataset.from_generator(load_raw_data, output_types=(tf.uint8, tf.uint8), output_shapes=(tf.TensorShape([None, 127]),
                                                                                                       tf.TensorShape([None, 127])))

data = data.unbatch()
# data = data.batch(3*60*150*midi_stuff.BASE_TICKS_PER_BEAT).shuffle(10000, reshuffle_each_iteration=True).unbatch()

size = 50000
data = data.batch(seq_length, drop_remainder=True).take(size).batch(BATCH_SIZE, drop_remainder=True)
train_data = data.skip(10)
test_data = data.take(10)



def build_model(batch_size=BATCH_SIZE):
    inputs = keras.layers.Input(batch_shape=(batch_size, None, 127), batch_size=batch_size)

    y = keras.layers.GRU(512, batch_size=batch_size, return_sequences=True, stateful=True, dropout=0.2)(inputs)
    y = keras.layers.Dropout(0.3)(y)
    y = keras.layers.Dense(127*len(vocab))(y)
    y = keras.layers.Reshape((-1, 127, len(vocab)))(y)

    m = keras.Model(inputs=inputs, outputs=y)
    m.summary()
    return m


def remove_checkpoints():
    dir_path = os.path.join(os.getcwd(), 'checkpoints2')
    try:
        shutil.rmtree(dir_path)
    except OSError as e:
        print("Error: %s : %s" % (dir_path, e.strerror))


class loss_callback(keras.callbacks.Callback):
    def __init__(self):
        super(loss_callback, self).__init__()

    def on_epoch_end(self, epoch, logs=None):
        global loss
        loss = logs['loss']


loss = -1
prev_loss = 100


def train(epochs=10, cont=False, lr=0.001, checkpoint=tf.train.latest_checkpoint(checkpoint_dir="checkpoints2")):
    model = build_model(BATCH_SIZE)

    ini_epoch = 0
    if cont:
        latest = checkpoint
        if latest is not None:
            ini_epoch = int(latest[17:])
            model.load_weights(latest)
    else:
        remove_checkpoints()

    def get_scheduler(ini_lr=lr):
        def schedule(epoch, l_r):
            global prev_loss
            limit = 0.015 * (1 / 3) ** (log(l_r / ini_lr) / log(0.5))
            if prev_loss == -1:
                if not cont:
                    l_r = ini_lr
            elif prev_loss - loss < limit:
                l_r *= 1 / 2
                limit *= 1 / 2
            prev_loss = loss

            print("Learning rate:", l_r)
            return l_r

        return keras.callbacks.LearningRateScheduler(schedule)

    model.compile(optimizer="adam",
                  loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics="accuracy", )
    checkpoint_call_back = tf.keras.callbacks.ModelCheckpoint("checkpoints2/ckpt_{epoch}", save_weights_only=True,
                                                              save_freq=3)

    model.fit(train_data,
              epochs=epochs + ini_epoch, initial_epoch=ini_epoch,
              callbacks=[get_scheduler(lr), checkpoint_call_back, loss_callback()],
              verbose=1,
              validation_data=test_data,
              validation_freq=3)
    return model


def generate_text(m, num, temperature):
    input_eval = np.array([[0]*127])
    input_eval = tf.expand_dims(input_eval, 0)

    text_generated = []

    m.reset_states()
    for i in range(num):
        predictions = m(input_eval)

        predictions = tf.squeeze(predictions)
        predictions = tf.map_fn(lambda x: tf.random.categorical(tf.expand_dims(x, 0)/temperature, num_samples=1).numpy()[-1, 0], predictions).numpy()

        input_eval = tf.expand_dims([predictions], 0)

        add = decode(predictions).tolist()
        text_generated.append(add)

    return text_generated


def guess(num=1000, temp=1, model=None,
          checkpoint=tf.train.latest_checkpoint(checkpoint_dir="./checkpoints2")):
    if model is None:
        model = build_model(1)
        model.load_weights(checkpoint).expect_partial()
    model.build(tf.TensorShape([1, None, 127]))
    generated = generate_text(model, num, temp)
    return generated


def save_whole_model(directory, checkpoint=tf.train.latest_checkpoint(checkpoint_dir="./checkpoints2")):
    model = build_model(1)
    model.load_weights(checkpoint).expect_partial()
    model.save(directory)


if __name__ == "__main__":
    train(3, cont=False)
    generated = guess(10)
    print(np.array(generated))

