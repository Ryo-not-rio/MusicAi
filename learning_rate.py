import numpy as np
import tensorflow.keras as keras
from tensorflow.keras import callbacks
from tensorflow.keras import backend as K
import seaborn as sns
import ai
import matplotlib.pyplot as plt
import pickle

class MultiplicativeLearningRate(callbacks.Callback):

    def __init__(self, factor):
        self.factor = factor
        self.notes = []
        self.vels = []
        self.times = []
        self.lengths = []
        self.learning_rates = []

    def on_batch_end(self, batch, logs):
        self.learning_rates.append(K.get_value(self.model.optimizer.lr))
        self.notes.append(logs["note_loss"])
        self.vels.append(logs["vel_loss"])
        self.lengths.append(logs["length_loss"])
        self.times.append(logs["time_loss"]/(0.9e5))
        K.set_value(self.model.optimizer.lr, self.model.optimizer.lr * self.factor)


def find_lr(model, dataset, min_lr, max_lr, batch_size):
    # Train for 1 epoch, starting with minimum learning rate and increase it
    # Compute learning rate multiplicative factor
    num_iter = 535
    lr_factor = np.exp(np.log(max_lr / min_lr) / num_iter)

    K.set_value(model.optimizer.lr, min_lr)
    lr_callback = MultiplicativeLearningRate(lr_factor)
    model.fit(dataset, epochs=1, batch_size=batch_size, callbacks=[lr_callback])

    # Plot loss vs log-scaled learning rate
    # plot = sns.lineplot(x=lr_callback.learning_rates, y=lr_callback.notes, label="notes")
    # sns.lineplot(x=lr_callback.learning_rates, y=lr_callback.vels, label="vels")
    # sns.lineplot(x=lr_callback.learning_rates, y=lr_callback.times, label="times")
    with open("lrs.pkl", "wb") as f:
        pickle.dump(lr_callback.learning_rates, f)
        pickle.dump(lr_callback.notes, f)
        pickle.dump(lr_callback.vels, f)
        pickle.dump(lr_callback.times, f)
        pickle.dump(lr_callback.lengths, f)
    # plot.set(xscale="log",
    #          xlabel="Learning Rate (log-scale)",
    #          ylabel="Training Loss",
    #          title="Optimal learning rate is slightly below minimum")
    # plt.legend()
    # plt.show()


if __name__ == "__main__":
    batch_size = ai.batch_size # Change in ai.py as well when changing this
    model = ai.build_model(batch_size)
    model.compile(optimizer="adam",
                  loss={"note": keras.losses.SparseCategoricalCrossentropy(from_logits=True, name="loss"),
                        "vel": keras.losses.SparseCategoricalCrossentropy(from_logits=True, name="loss"),
                        "time": "mse",
                        "length": keras.losses.SparseCategoricalCrossentropy(from_logits=True, name="loss")},
                  loss_weights={"note": 1e4, "vel": 1e4, "time": 1, "length": 1e4},
                  metrics={"note": "accuracy", "vel": "accuracy",
                           "time": [keras.metrics.RootMeanSquaredError(name="rms")],
                           "length": "accuracy"})

    find_lr(model, ai.train_data, 10 ** -4, 1e4, batch_size=batch_size)

    with open("lrs.pkl", "rb") as f:
        rates = pickle.load(f)
        notes = pickle.load(f)
        vels = pickle.load(f)
        times = pickle.load(f)
        lengths = pickle.load(f)

    # Plot loss vs log-scaled learning rate
    plot = sns.lineplot(x=rates, y=notes, label="notes")

    plot.set(xscale="log",
             xlabel="Learning Rate (log-scale)",
             ylabel="Training Loss",
             title="Optimal learning rate is slightly below minimum")
    plt.legend()
    plt.show()

    plot = sns.lineplot(x=rates, y=notes, label="vels")
    plot.set(xscale="log",
             xlabel="Learning Rate (log-scale)",
             ylabel="Training Loss",
             title="Optimal learning rate is slightly below minimum")
    plt.legend()
    plt.show()

    plot = sns.lineplot(x=rates, y=notes, label="times")
    plot.set(xscale="log",
             xlabel="Learning Rate (log-scale)",
             ylabel="Training Loss",
             title="Optimal learning rate is slightly below minimum")
    plt.legend()
    plt.show()

    plot = sns.lineplot(x=rates, y=notes, label="lengths")
    plot.set(xscale="log",
             xlabel="Learning Rate (log-scale)",
             ylabel="Training Loss",
             title="Optimal learning rate is slightly below minimum")
    plt.legend()
    plt.show()