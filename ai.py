import tensorflow as tf
from tensorflow import keras
import numpy as np
from sklearn.model_selection import train_test_split
import os
import shutil

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"


def to_index(arr):
    return np.array([note2idx[arr[0]], vel2idx[arr[1]], arr[2]])


def decode(arr):
    return np.array([idx2notes[arr[0]], idx2vel[arr[1]], arr[2]])


def split_input(chunk):
    return chunk[:-1], chunk[1:]


with open('data.npy', 'rb') as f:
    data = np.load(f)

notes = sorted(set(data[:, 0]))
vels = sorted(set(data[:, 1]))
times = sorted(set(data[:, 2]))
note2idx = {i: k for k, i in enumerate(notes)}
idx2notes = np.array(notes)
vel2idx = {i: k for k, i in enumerate(vels)}
idx2vel = np.array(vels)
time2idx = {i: i for k, i in enumerate(times)}
idx2time = np.array(times)

indexed = np.array(list(map(lambda x: to_index(x), data)))
seq_length = 100 ###### UNCOMMENT BELOW WHEN CHANGING THIS!!!!!!

#########################################################################

X, note_y, vel_y, time_y = [], [], [], []

for i in range(0, len(indexed) - seq_length - 1, int(seq_length * 0.1)):
    temp_x, temp_y = split_input(indexed[i:i + seq_length + 1])
    X.append(temp_x)
    note_y.append(temp_y[:, 0])
    vel_y.append(temp_y[:, 1])
    time_y.append(temp_y[:, 2])
X, note_y, vel_y, time_y = np.array(X), np.array(note_y), np.array(vel_y), np.array(time_y)

with open("cat_datasets.npy", 'wb') as f:
    np.save(f, X)
    np.save(f, note_y)
    np.save(f, vel_y)
    np.save(f, time_y)

################################################################################

BATCH_SIZE = 4

with open("cat_datasets.npy", "rb") as f:
    X = np.load(f)
    note_y = np.load(f)
    vel_y = np.load(f)
    time_y = np.load(f)

(trainX, valX, trainNoteY, testNoteY,
 trainVelY, testVelY, trainTimeY, testTimeY) = train_test_split(X, note_y, vel_y, time_y,
                                                                test_size=(650 // BATCH_SIZE) * BATCH_SIZE)

size = BATCH_SIZE * (len(trainX) // BATCH_SIZE)
trainX = trainX[:size]
valX = valX[:size]
trainNoteY = trainNoteY[:size]
testNoteY = testNoteY[:size]
trainVelY = trainVelY[:size]
testVelY = testVelY[:size]
trainTimeY = trainTimeY[:size]
testTimeY = testTimeY[:size]

embed_dim = 30
embed_dim2 = 1


def build_model(batch_size):
    inputs = keras.layers.Input(batch_shape=(batch_size, None, 3), batch_size=batch_size)
    note_x, vel_x, time_x = tf.split(inputs, 3, -1)

    note_x1 = keras.layers.Embedding(len(notes) + 1, embed_dim)(note_x)
    vel_x1 = keras.layers.Embedding(len(vels) + 1, embed_dim)(vel_x)
    note_x1 = keras.layers.Reshape((-1, embed_dim))(note_x1)
    vel_x1 = keras.layers.Reshape((-1, embed_dim))(vel_x1)
    time_x1 = keras.layers.Reshape((-1, 1))(time_x)

    x1 = keras.layers.concatenate((note_x1, vel_x1, time_x1), -1)
    x1 = keras.layers.Dense(64)(x1)

    note_x2 = keras.layers.Embedding(len(notes) + 1, embed_dim2)(note_x)
    vel_x2 = keras.layers.Embedding(len(vels) + 1, embed_dim2)(vel_x)
    note_x2 = keras.layers.Reshape((-1, embed_dim2))(note_x2)
    vel_x2 = keras.layers.Reshape((-1, embed_dim2))(vel_x2)
    time_x2 = keras.layers.Reshape((-1, 1))(time_x)

    x2 = keras.layers.concatenate((note_x2, vel_x2, time_x2), -1)
    x2 = keras.layers.Dense(64)(x2)

    y1 = tf.keras.layers.LSTM(512, batch_size=batch_size, return_sequences=True, stateful=True,
                              recurrent_initializer='glorot_uniform', dropout=0.01)(x1)
    y2 = tf.keras.layers.LSTM(400, batch_size=batch_size, return_sequences=True, stateful=True,
                              recurrent_initializer='glorot_uniform', dropout=0.01)(x1)
    y3 = tf.keras.layers.LSTM(1024, batch_size=batch_size, return_sequences=True, stateful=True,
                              recurrent_initializer='glorot_uniform', dropout=0.5, kernel_regularizer="l1_l2",
                              recurrent_regularizer="l1_l2")(x2)

    y_1 = keras.layers.Dense(len(notes), activation=None, name="note_output")(y1)
    y_2 = keras.layers.Dense(len(vels), activation=None, name="vel_output")(y2)
    y_3 = keras.layers.Dense(1, activation="relu", name="time_output")(y3)

    m = keras.Model(inputs=inputs, outputs=[y_1, y_2, y_3])
    return m


def loss(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)


def rmse(y_true, y_pred):
    return tf.math.sqrt(keras.losses.mean_squared_error(y_true, y_pred))


def remove_checkpoints():
    dir_path = os.path.join(os.getcwd(), 'checkpoints')
    try:
        shutil.rmtree(dir_path)
    except OSError as e:
        print("Error: %s : %s" % (dir_path, e.strerror))


def train(epochs=10, cont=False, checkpoint=tf.train.latest_checkpoint(checkpoint_dir="checkpoints")):
    model = build_model(BATCH_SIZE)
    ini_epoch = 0
    model.compile(optimizer="adam",
                  loss={"note_output": loss,
                        "vel_output": loss,
                        "time_output": "mse"},
                  loss_weights={"note_output": 1, "vel_output": 1, "time_output": 1},
                  metrics={"note_output": "accuracy", "vel_output": "accuracy", "time_output": [keras.metrics.RootMeanSquaredError(), "mae"]})
    checkpoint_call_back = tf.keras.callbacks.ModelCheckpoint("checkpoints/ckpt_{epoch}", save_weights_only=True)

    if cont:
        latest = checkpoint
        if latest is not None:
            ini_epoch = int(latest[17:])
            model.load_weights(latest)
    else:
        remove_checkpoints()

    model.fit(x=trainX,
              y={"note_output": trainNoteY, "vel_output": trainVelY, "time_output": trainTimeY},
              batch_size=BATCH_SIZE, epochs=epochs+ini_epoch, initial_epoch=ini_epoch, callbacks=[checkpoint_call_back],
              verbose=2,
              validation_data=(valX, {"note_output": testNoteY, "vel_output": testVelY, "time_output": testTimeY}),
              validation_freq=3)
    return model


def generate_text(m, start_string, num, temperature):
    input_eval = np.array([[to_index(x) for x in start_string]])
    input_eval = tf.expand_dims(input_eval, 0)

    text_generated = []

    m.reset_states()
    for i in range(num):
        predictions = m(input_eval)
        note_predict, vel_predict, time_predict = predictions[0], predictions[1], predictions[2]
        note_predict, vel_predict, time_predict = tf.squeeze(note_predict, 0), tf.squeeze(vel_predict, 0), \
                                                  tf.squeeze(time_predict, 0).numpy()[0, 0]
        note_predict, vel_predict = note_predict/temperature, vel_predict/temperature
        note_id, vel_id = tf.random.categorical(note_predict, num_samples=1).numpy()[0, 0], \
                          tf.random.categorical(vel_predict, num_samples=1).numpy()[0, 0]

        if time_predict < 0:
            time_predict = 0

        input_eval = tf.expand_dims(np.array([note_id, vel_id, time_predict]), 0)

        text_generated.append(decode([int(note_id), int(vel_id), int(time_predict)]).tolist())

    return [start_string[0].tolist()] + text_generated


def guess(start=[[76, 60, 240]], num=1000, temp=1, checkpoint=tf.train.latest_checkpoint(checkpoint_dir="checkpoints")):
    model = build_model(1)
    model.load_weights(checkpoint).expect_partial()
    model.build(tf.TensorShape([1, None, 3]))
    return generate_text(model, np.array(start), num, temp)
