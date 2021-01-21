import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
import numpy as np
import shutil
from math import log
import pickle

try:
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
except:
    print("No GPU")


def to_index(arr):
    return np.array([note2idx[x.encode()] for x in arr])


def decode(arr):
    return np.array([idx2note[int(x)].decode() for x in arr])


def split_input(chunk):
    return chunk[:-1], chunk[1:]


def load_raw_data():
    directory = os.fsencode("data2")

    for file in os.listdir(directory):
        data = []
        file_name = os.fsdecode(file)
        if file_name[-3:] == "npy":
            data.append(["-1"]*127)
            with(open(os.path.join(os.fsdecode(directory), file_name), "rb")) as f:
                data.append(np.load(f))

        arr = np.vstack(np.array(data))
        arr = np.array(list(map(to_index, arr)))
        for i in range(arr.shape[0]-1):
            yield arr[i], arr[i+1]


def get_vocab(file_name, new_data=False):
    try:
        with open(file_name, "rb") as f:
            vocab = pickle.load(f)
    except FileNotFoundError:
        vocab = [b"-1"]
        new_data = True

    if new_data:
        directory = os.fsencode("data2")
        for file in os.listdir(directory):
            file_name_temp = os.fsdecode(file)
            if file_name_temp[-3:] == "npy":
                with(open(os.path.join(os.fsdecode(directory), file_name_temp), "rb")) as f:
                    for d in set(np.load(f).flatten()):
                        if d not in vocab:
                            vocab.append(d.encode())

    with open(file_name, "wb") as f:
        pickle.dump(vocab, f)

    return vocab


seq_length = 100
BATCH_SIZE = 4

vocab = get_vocab("matrix_vocab.pkl")

note2idx = {i: k for k, i in enumerate(vocab)}
idx2note = np.array(vocab)

data = tf.data.Dataset.from_generator(load_raw_data, output_types=(tf.uint8, tf.uint8), output_shapes=(tf.TensorShape([127]),
                                                                                                       tf.TensorShape([127])))

data = data.batch(seq_length, drop_remainder=True)

train_data = data.skip(BATCH_SIZE * 50).batch(BATCH_SIZE, drop_remainder=True)
test_data = data.take(BATCH_SIZE * 50).batch(BATCH_SIZE, drop_remainder=True)



def build_model(batch_size=BATCH_SIZE):
    inputs = keras.layers.Input(batch_shape=(batch_size, None, 127), batch_size=batch_size)

    y = keras.layers.Dropout(0.1)(inputs)

    y = keras.layers.GRU(512, batch_size=batch_size, return_sequences=True, stateful=True, dropout=0.2)(y)

    y = keras.layers.Dense(127*len(vocab))(y)
    y = keras.layers.Reshape((-1, 127, len(vocab)))(y)

    m = keras.Model(inputs=inputs, outputs=y)
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
              verbose=2,
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
        predictions = tf.map_fn(lambda x: tf.random.categorical(tf.expand_dims(x, 0), num_samples=1).numpy()[-1, 0], predictions).numpy()

        input_eval = tf.expand_dims([predictions], 0)

        add = decode(predictions).tolist()
        text_generated.append(add)
        # if -1 not in add:
        #     text_generated.append(add)
        # else:
        #     num += 1

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