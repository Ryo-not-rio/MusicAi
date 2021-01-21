import ai
import midi_stuff
import tensorflow as tf
import preprocess


def make_midi(num, model=None):
    generated = ai.guess(num=num,
                         start=[[76, 60, 240, 227], [52, 48, 0, 227], [83, 66, 240, 1481], [59, 52, 0, 1481]],
                         temp=0.8,
                         model=model)
    midi_stuff.make_midi_file(generated)


if __name__ == "__main__":
    preprocess.process()
    # # ai.train(100, cont=False)
    # # ai.save_whole_model("models/model_acc_working")
    # m = ai.build_model(1)
    # # new_model = tf.keras.models.load_model('models/model_acc_working')
    # make_midi(2000, m)
