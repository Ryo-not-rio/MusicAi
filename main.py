import ai2_classed
import ai1_classed
import ai3
import ai3_with_vel
import mido
import Ai_vel
from tensorflow import keras as keras
import pickle


def make_midi(num, ai_obj, model=None, checkpoint_num=None, vel_model=None, vocabs=None, file="new.mid"):
    if vel_model is None:
        generated = ai_obj.guess(num=num,
                                 temp=0.9,
                                 model=model,
                                 checkpoint_num=checkpoint_num)
    else:
        generated = ai_obj.guess(num=num,
                                 temp=0.9,
                                 model=model,
                                 vel_model=vel_model,
                                 checkpoint_num=checkpoint_num,
                                 vocabs=vocabs)
    notes = ai.data_to_midi_sequence(generated, vocabs=vocabs)
    ai_obj.make_midi_file(notes, file)


if __name__ == "__main__":
    ai = ai3.Ai3()
    # ai.process_all()
    # converted = ai.midi_to_data(mido.MidiFile("midis/alb_esp1.mid"), ai.vocabs)
    # print(converted)

    # ai.train(46, cont=True)
    # ai.save_whole_model("vel_models", "categorical_model")
    with open("models/Ai3_no_vels.pkl", "rb") as f:
        vocabs = pickle.load(f)

    make_midi(1000, ai, model=keras.models.load_model("models/Ai3_no_vels"),
              vel_model=keras.models.load_model("vel_models/categorical_model"), file="temp.mid", vocabs=vocabs)
    # new_model = tf.keras.models.load_model('models/model_acc_working')
    # make_midi(1000, ai)

