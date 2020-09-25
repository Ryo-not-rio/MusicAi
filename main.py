import ai2
import ai
import file_to_npy
import midi_stuff


def make_midi(num):
    generated = ai.guess(num=num,
                         start=[[76, 60, 240], [76, 0, 227], [83, 66, 13], [83, 0, 1481]],
                         temp=1.1,
                         checkpoint="checkpoints/ckpt_4")
    midi_stuff.make_midi_file(generated)


if __name__ == "__main__":
    # m = ai.train(0, True, checkpoint="checkpoints/ckpt_4")
    # m.save("models/model_best_2")
    make_midi(2000)
