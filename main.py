import ai2
# import ai
import midi_stuff
import preprocess
import tensorflow as tf


def make_midi(num, model=None):
    generated = ai2.guess(num=num,
                          temp=0.01,
                          model=model)
    midi_stuff.make_midi_file(generated)


if __name__ == "__main__":
    preprocess.process()
    ai2.train(100, cont=False)
    # ai2.save_whole_model("models/model_acc_working")

    # new_model = tf.keras.models.load_model('models/model_acc_working')
    make_midi(10000)

