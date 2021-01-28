import ai2_classed
import ai1_classed
import ai3
import mido
import tensorflow as tf


def make_midi(num, ai_obj, model=None, checkpoint_num=None):
    generated = ai_obj.guess(num=num,
                             temp=1.1,
                             model=model,
                             checkpoint_num=checkpoint_num)
    notes = ai.data_to_midi_sequence(generated)
    ai_obj.make_midi_file(notes)


if __name__ == "__main__":
    ai = ai1_classed.Ai1()
    # ai.process_all()
    # converted = ai.midi_to_data(mido.MidiFile("midis/alb_esp1.mid"), ai.vocabs)
    # print(converted)
    # ai.process_all()
    # ai.train(100, cont=False)

    new_model = tf.keras.models.load_model('models/model_acc_working')
    new_model.summary()
    # make_midi(1000, ai)

