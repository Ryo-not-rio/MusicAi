import ai2
import ai
import file_to_npy
import midi_stuff


def make_midi(num):
    generated = ai.guess(num=num,
                         start=[[76, 60, 240, 227], [52, 48, 0, 227], [83, 66, 240, 1481], [59, 52, 0, 1481]],
                         temp=1.1)
    print(generated)
    midi_stuff.make_midi_file(generated)


if __name__ == "__main__":
    ai.train(1, cont=False) # Uncomment to train
    # m = ai.train(0, True, checkpoint="checkpoints/ckpt_4") # Uncomment to train from particular checkpoint
    # m.save("models/model_best_2") # Uncomment to save whole model
    make_midi(10) # Uncomment to create new music
