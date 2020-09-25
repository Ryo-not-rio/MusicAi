import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
import shutil
import random


physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

seq_length = 100

BATCH_SIZE = 64
BUFFER_SIZE = 50000
rnn_units = 128


def split_input(chunk):
    return chunk[:-1], chunk[1:]


def build_model(batch_size):
    m = keras.Sequential()
    m.add(keras.layers.Input(batch_input_shape=[batch_size, None, 3]))
    m.add(keras.layers.Dense(128, activation="sigmoid"))
    m.add(keras.layers.LSTM(rnn_units, return_sequences=True, stateful=True, recurrent_initializer='glorot_uniform', dropout=0.05))
    m.add(keras.layers.LSTM(rnn_units, return_sequences=True, stateful=True, recurrent_initializer='glorot_uniform', dropout=0.05))
    m.add(keras.layers.Dense(128, activation="relu"))
    m.add(keras.layers.Dense(3))
    return m


def loss(labels, logits):
    # best - mean_absolute_error, Huber
    return tf.keras.losses.mean_absolute_error(labels, logits)


def remove_checkpoints():
    dir_path = os.path.join(os.getcwd(), 'checkpoints')

    try:
        shutil.rmtree(dir_path)
    except OSError as e:
        print("Error: %s : %s" % (dir_path, e.strerror))


def train(epochs=10, cont=False):
    with open('data.npy', 'rb') as f:
        data = np.load(f)

    dataset = tf.data.Dataset.from_tensor_slices(data)
    sequences = dataset.batch(seq_length + 1, drop_remainder=True)
    dataset = sequences.map(split_input)
    dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
    test_dataset = dataset.take(3)
    train_dataset = dataset.skip(3)

    model = build_model(BATCH_SIZE)
    ini_epoch = 0
    model.compile(optimizer='adam', loss=tf.keras.losses.Huber(), metrics=['accuracy'])
    checkpoint_call_back = tf.keras.callbacks.ModelCheckpoint("./checkpoints/ckpt_{epoch}", save_weights_only=True,
                                                              save_freq='epoch')
    if cont:
        latest = tf.train.latest_checkpoint(checkpoint_dir="checkpoints")
        if latest is not None:
            ini_epoch = int(latest[17:])
            model.load_weights(latest)
    else:
        remove_checkpoints()
    model.fit(train_dataset, epochs=epochs+ini_epoch, validation_data=test_dataset, validation_freq=3,
              callbacks=[checkpoint_call_back], initial_epoch=ini_epoch, verbose=2)


integerize = np.vectorize(lambda x: int(x))


def generate_text(m, start_string, num_generate=1000, randomness=0):
    input_eval = tf.expand_dims(start_string, 0)
    generated = []

    # low -> high == predictable -> surprising
    temperature = 1
    m.reset_states()
    # m.summary()
    for i in range(num_generate):
        predictions = m(input_eval)

        predictions = tf.squeeze(predictions, 0)

        predicted_id = predictions[0].numpy()

        if predicted_id[0] < 0:
            predicted_id[0] = 0
        elif predicted_id[0] > 127:
            predicted_id[0] = 127
        if predicted_id[1] < 40:
            predicted_id[1] = 0
        elif predicted_id[1] > 127:
            predicted_id[1] = 127
        if predicted_id[2] < 0:
            predicted_id[2] = 0

        generated.append(integerize(predicted_id))

        for x, y in enumerate(predicted_id):
            predicted_id[x] = y * random.randint(100 - randomness, 100 + randomness) * 0.01
        input_eval = tf.expand_dims([predicted_id], 0)

    return list([start_string[0]]) + list(generated)


def guess(start=[[76, 60, 240]], num=1000, randomness=0):
    model = build_model(1)
    model.load_weights(tf.train.latest_checkpoint(checkpoint_dir="checkpoints")).expect_partial()
    model.build(tf.TensorShape([1, None, 3]))
    return generate_text(model, np.array(start), num, randomness)
