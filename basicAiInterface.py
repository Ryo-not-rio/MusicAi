from abc import ABCMeta, abstractmethod
import os
import shutil
from tensorflow import keras
import tensorflow as tf
from math import log
import numpy as np
from typing import Union, Optional


class Interface:
    __metaclass__ = ABCMeta

    @staticmethod
    def remove_checkpoints(dir):
        try:
            shutil.rmtree(dir)
        except OSError as e:
            print("Error: %s : %s" % (dir, e.strerror))

    def __init__(self, check_dir: str, data_dir: str, batch_size: int) -> None:
        self.check_dir = check_dir
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.loss = -1
        self.prev_loss = 100
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
            self.process_all()

    @abstractmethod
    def process_all(self, midi_dir: str = "midis") -> list: raise NotImplementedError

    # noinspection PyMissingConstructor
    class LossCallback(keras.callbacks.Callback):
        def __init__(self, outer_class):
            super(outer_class.LossCallback, self).__init__()
            self.outer = outer_class

        def on_epoch_end(self, epoch, logs=None):
            self.outer.loss = logs['loss']

    def loss_callback(self):
        return self.LossCallback(self)

    def get_scheduler(self, ini_lr: float, cont: bool) -> keras.callbacks.LearningRateScheduler:
        def schedule(epoch: int, l_r: float):
            limit = 0.015 * (1 / 3) ** (log(l_r / ini_lr) / log(0.5))
            if self.prev_loss == -1:
                if not cont:
                    l_r = ini_lr
            elif self.prev_loss - self.loss < limit:
                l_r *= 1 / 2
                limit *= 1 / 2
            self.prev_loss = self.loss

            print("Learning rate:", l_r)
            return l_r

        return keras.callbacks.LearningRateScheduler(schedule)

    def get_checkpoint(self, checkpoint_num: int) -> str:
        if checkpoint_num is not None:
            return os.path.join(self.check_dir, f"ckpt_{checkpoint_num}")

        ret = tf.train.latest_checkpoint(self.check_dir)
        if ret is None:
            print("here")
        return ret

    @abstractmethod
    def get_dataset(self) -> Union[np.array, tf.data.Dataset]: raise NotImplementedError

    @abstractmethod
    def build_model(self, batch_size: int) -> keras.Model: raise NotImplementedError

    @abstractmethod
    def train(self, epochs=10, cont=False, lr=0.001, checkpoint_num=None): raise NotImplementedError

    def save_whole_model(self, save_dir, model_name, checkpoint_num=None):
        checkpoint = self.get_checkpoint(checkpoint_num)
        model = self.build_model(1)
        model.load_weights(checkpoint).expect_partial()
        model.save(os.path.join(save_dir, model_name))
