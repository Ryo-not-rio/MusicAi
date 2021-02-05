from abc import ABCMeta, abstractmethod
import os
import shutil
from tensorflow import keras
import tensorflow as tf
from math import log
import pickle
import mido
import numpy as np
from typing import Union
import time
from basicAiInterface import Interface


class AiInterface(Interface):
    __metaclass__ = ABCMeta

    def __init__(self, check_dir: str, data_dir: str, vocab_file: str, batch_size: int, ticks_per_beat: int) -> None:
        super().__init__(check_dir, data_dir, batch_size)
        self.ticks_per_beat = ticks_per_beat
        self.vocab_file = vocab_file

        try:
            with open(vocab_file, "rb") as f:
                self.vocabs = pickle.load(f)
        except FileNotFoundError:
            self.process_all() # Sets self.vocabs

    def make_midi_file(self, notes, file="new.mid"):
        new_mid = mido.MidiFile()
        new_mid.ticks_per_beat = self.ticks_per_beat
        new_track = mido.MidiTrack()
        new_mid.tracks.append(new_track)

        for n in notes:
            try:
                new_track.append(mido.Message('note_on', note=n[0], velocity=n[1], time=n[2]))
            except ValueError as e:
                print(e, " for ", n)

        new_track.append(mido.MetaMessage('end_of_track', time=1))

        new_mid.save(file)
        print("successfully saved midi file")

    @abstractmethod
    def midi_to_data(self, midi: mido.MidiFile, vocabs: list) -> (np.array, list): raise NotImplementedError

    @abstractmethod
    def data_to_midi_sequence(self, list) -> list: raise NotImplementedError

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
        print("time_taken: ", time.time()-start)
        return vocabs

    @abstractmethod
    def get_dataset(self) -> Union[np.array, tf.data.Dataset]: raise NotImplementedError

    @abstractmethod
    def build_model(self, batch_size) -> keras.Model: raise NotImplementedError

    @abstractmethod
    def train(self, epochs=10, cont=False, lr=0.001, checkpoint_num=None): raise NotImplementedError

    @abstractmethod
    def generate_text(self, model, num, start, temperature) -> list: raise NotImplementedError

    @abstractmethod
    def guess(self, num, start, temp, model, checkpoint_num) -> list: raise NotImplementedError

    def save_whole_model(self, save_dir, model_name, checkpoint_num=None):
        checkpoint = self.get_checkpoint(checkpoint_num)
        model = self.build_model(1)
        model.load_weights(checkpoint).expect_partial()
        model.save(os.path.join(save_dir, model_name))
        with open(os.path.join(save_dir, model_name+".pkl"), "wb") as f:
            pickle.dump(self.vocabs, f)
