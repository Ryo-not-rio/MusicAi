"""
Handles all the machine learning aspects
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
import shutil

### Configuring gpu for training
try:
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
except:
    print("No GPU")


def to_index(arr):
    return np.array([note2idx[arr[0]], vel2idx[arr[1]], time2idx[arr[2]], length2idx[arr[3]]])


def decode(arr):
    return np.array([idx2note[arr[0]], idx2vel[arr[1]], idx2time[arr[2]], idx2length[arr[3]]])


def split_input(chunk):
    return chunk[:-1], chunk[1:]

"""
Load data from the data folder and put it in a single list
"""
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

def cut(d):
    length = d.shape[0] * (d.shape[0] // (seq_length*BATCH_SIZE))
    return d[:length]


"""
Get data from data folder, process it using the passed function,
concatenate the result.
"""
def process_data(function):
    data = load_raw_data()
    data = np.array(list(map(function, data)))
    data = np.vstack(data)
    return data


"""
No processing, just straight data
"""
def straight_data():
    return np.vstack(load_raw_data())

seq_length = 100 # Length of sequence the LSTM looks at
BATCH_SIZE = 5

data = process_data(cut)

# All the possible values i.e. vocabulary for each data item. 
notes = sorted(set(data[:, 0]))
vels = sorted(set(data[:, 1]))
times = sorted(set(data[:, 2]))
lengths = sorted(set(data[:, 3]))

"""
The vocabulary but the value and index are swapped 
so the index can be retrieved by passing the value
being searched for.
"""
note2idx = {i: k for k, i in enumerate(notes)}
idx2note = np.array(notes)
vel2idx = {i: k for k, i in enumerate(vels)}
idx2vel = np.array(vels)
time2idx = {i: k for k, i in enumerate(times)}
idx2time = np.array(times)
length2idx = {i: k for k, i in enumerate(lengths)}
idx2length = np.array(lengths)

# Changing all the data values to indexes within the vocabulary
indexed = np.array(list(map(lambda x: to_index(x), data)))

"""
Creating a tensorflow dataset.
X value - list of notes where each note is represented as [note, velocity, time, length]
y value - Dictionary containing a list of the correct result for each attribute.
"""
X, temp_y = indexed[:-1], indexed[1:]
note_y = temp_y[:, 0]
vel_y = temp_y[:, 1]
time_y = temp_y[:, 2]
length_y = temp_y[:, 3]
data = tf.data.Dataset.from_tensor_slices((X, {"note": note_y, "vel": vel_y, "time": time_y, "length": length_y}))

data = data.repeat(3).batch(seq_length, drop_remainder=True) # Grouping the dataset into the specified sequence length

# Shuffle and split the dataset so each full music sequence won't be separated
train_data = data.batch(BATCH_SIZE).shuffle(1000, reshuffle_each_iteration=True).unbatch().skip(BATCH_SIZE*2).batch(BATCH_SIZE, drop_remainder=True)
test_data = data.take(BATCH_SIZE*2).batch(BATCH_SIZE, drop_remainder=True)

embed_dim = 30


# Building the model using the functional API
def build_model(batch_size):
    inputs = keras.layers.Input(batch_shape=(batch_size, None, 4), batch_size=batch_size)
    note_x, vel_x, time_x, length_x = tf.split(inputs, 4, -1)
    note_x1 = keras.layers.Embedding(len(notes) + 1, embed_dim)(note_x)
    note_x1 = keras.layers.Reshape((-1, embed_dim))(note_x1)
    note_x1 = keras.layers.LSTM(64, batch_size=batch_size, return_sequences=True, stateful=True, dropout=0.1)(note_x1)

    vel_x1 = keras.layers.Embedding(len(vels) + 1, embed_dim)(vel_x)
    vel_x1 = keras.layers.Reshape((-1, embed_dim))(vel_x1)
    vel_x1 = keras.layers.LSTM(64, batch_size=batch_size, return_sequences=True, stateful=True, dropout=0.1)(vel_x1)

    time_x1 = keras.layers.Embedding(len(times) + 1, embed_dim)(time_x)
    time_x1 = keras.layers.Reshape((-1, embed_dim))(time_x1)
    time_x1 = keras.layers.LSTM(64, batch_size=batch_size, return_sequences=True, stateful=True, dropout=0.1)(time_x1)

    length_x1 = keras.layers.Embedding(len(lengths) + 1, embed_dim)(length_x)
    length_x1 = keras.layers.Reshape((-1, embed_dim))(length_x1)
    length_x1 = keras.layers.LSTM(64, batch_size=batch_size, return_sequences=True, stateful=True, dropout=0.1)(length_x1)

    y1 = keras.layers.concatenate((note_x1, vel_x1, time_x1, length_x1), -1)
    # y1 = keras.layers.BatchNormalization()(y1)

    y1 = keras.layers.LSTM(512, batch_size=batch_size, return_sequences=True, stateful=True, dropout=0.2)(y1)
    y1 = keras.layers.BatchNormalization()(y1)

    #### Performs better without this dense layer
    # y1 = keras.layers.Dense(64, activation="tanh")(y1)
    # y2 = keras.layers.Dense(64, activation="tanh")(y1)
    # y3 = keras.layers.Dense(64, activation="tanh")(y1)
    # y4 = keras.layers.Dense(64, activation="tanh")(y1)

    y_1 = keras.layers.Dense(len(notes), name="note")(y1)
    y_2 = keras.layers.Dense(len(vels), name="vel")(y1)
    y_3 = keras.layers.Dense(len(times), name="time")(y1)
    y_4 = keras.layers.Dense(len(lengths), name="length")(y1)

    m = keras.Model(inputs=inputs, outputs=[y_1, y_2, y_3, y_4])
    return m


def remove_checkpoints():
    dir_path = os.path.join(os.getcwd(), 'checkpoints')
    try:
        shutil.rmtree(dir_path)
    except OSError as e:
        print("Error: %s : %s" % (dir_path, e.strerror))


# Callback to reset the state of LSTM after each batch
class reset_callback(keras.callbacks.Callback):
  def __init__(self):
        super(reset_callback, self).__init__()

  def on_train_batch_begin(self, batch, logs=None):
    self.model.reset_states()


# Training
def train(epochs=10, cont=False, lr=10.0, checkpoint=tf.train.latest_checkpoint(checkpoint_dir="checkpoints")):
    model = build_model(BATCH_SIZE)

    # Reloading checkpoint if cont is set to True
    ini_epoch = 0
    if cont:
        latest = checkpoint
        if latest is not None:
            ini_epoch = int(latest[17:])
            model.load_weights(latest)
    else:
        remove_checkpoints()

    # Dynamically creating the scheduler to decrease epoch
    def get_scheduler(ini_lr=lr):
        def schedule(epoch, l_r):
            dec_epoch = 10
            border = dec_epoch

            if epoch < border:
                l_r = ini_lr
            elif epoch % dec_epoch == 0:
                l_r *= 1 / 3

            print("Learning rate:", l_r)
            return l_r

        return keras.callbacks.LearningRateScheduler(schedule)

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                  loss={"note": keras.losses.SparseCategoricalCrossentropy(from_logits=True, name="loss"),
                        "vel": keras.losses.SparseCategoricalCrossentropy(from_logits=True, name="loss"),
                        "time": keras.losses.SparseCategoricalCrossentropy(from_logits=True, name="loss"),
                        "length": keras.losses.SparseCategoricalCrossentropy(from_logits=True, name="loss"), },
                  metrics={"note": keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
                           "vel": keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
                           "time": keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
                           "length": keras.metrics.SparseCategoricalAccuracy(name="accuracy"), }, )
    checkpoint_call_back = tf.keras.callbacks.ModelCheckpoint("checkpoints/ckpt_{epoch}", save_weights_only=True,
                                                              save_freq=3)

    model.fit(train_data,
              epochs=epochs + ini_epoch, initial_epoch=ini_epoch,
              callbacks=[get_scheduler(lr), checkpoint_call_back, reset_callback()],
              verbose=2,
              validation_data=test_data,
              validation_freq=3)
    return model


"""
Generating a sequence of notes by taking starting notes
and passing the result back into the model
"""
def generate_text(m, start_string, num, temperature):
    input_eval = np.array([[to_index(x) for x in start_string]]) # Formatting the start string
    input_eval = tf.expand_dims(input_eval, 0)

    text_generated = []

    m.reset_states()
    for i in range(num):
        predictions = m(input_eval) # Get the predicted probability distributions for each attribute
        note_predict, vel_predict, time_predict, length_predict = predictions[0], predictions[1], predictions[2], predictions[3] # split prediction
        note_predict, vel_predict, time_predict, length_predict = tf.squeeze(note_predict, 0), tf.squeeze(vel_predict,0), \
                                                                  tf.squeeze(time_predict, 0), tf.squeeze(length_predict, 0)
        note_predict, vel_predict, time_predict, length_predict = note_predict / temperature, vel_predict / temperature, time_predict / temperature, length_predict / temperature
        note_id, vel_id, time_predict, length_predict = tf.random.categorical(note_predict, num_samples=1).numpy()[-1, 0], \
                                                        tf.random.categorical(vel_predict, num_samples=1).numpy()[-1, 0], \
                                                        tf.random.categorical(time_predict, num_samples=1).numpy()[-1, 0], \
                                                        tf.random.categorical(length_predict, num_samples=1).numpy()[-1, 0]

        # Preparing the result to be fed back to the model
        input_eval = tf.expand_dims(np.array([note_id, vel_id, time_predict, length_predict]), 0)

        text_generated.append(decode([int(note_id), int(vel_id), int(time_predict), int(length_predict)]).tolist())

    return [x.tolist() for x in start_string] + text_generated



# Creating music
def guess(start=((76, 60, 240, 277)), num=1000, temp=1,
          checkpoint=tf.train.latest_checkpoint(checkpoint_dir="checkpoints")):
    model = build_model(1)
    model.load_weights(checkpoint).expect_partial()
    model.build(tf.TensorShape([1, None, 3]))
    generated = generate_text(model, np.array(start), num, temp)
    return generated


if __name__ == "__main__":
    train(100, cont=False)