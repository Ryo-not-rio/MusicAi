"""
Handling all the processing of the midi files
"""
import mido
import os
import ast
import numpy as np

def make_midi_file(notes, ticks_per_beat, file="new.mid"):
    new_mid = mido.MidiFile()
    new_mid.ticks_per_beat = ticks_per_beat
    new_track = mido.MidiTrack()
    new_mid.tracks.append(new_track)

    for n in notes:
        try:
            new_track.append(mido.Message('note_on', note=n[0], velocity=n[1], time=n[2]))
        except ValueError as e:
            print(e)
            print(n)

    new_track.append(mido.MetaMessage('end_of_track', time=1))

    new_mid.save(file)
    print("successfully saved midi file")



# Unformatting each note to the format [note, velocity, time]
def unconvert(notes):
    i = 0
    while i < len(notes):
        data = notes[i]
        note, vel, time = data[0], data[1], data[2]
        data[2] = round(time * BASE_TICKS_PER_BEAT)

        if vel != 0:
            length = data[3]
            insert_ind = i+1
            while insert_ind < len(notes) and length > notes[insert_ind][2]:
                length -= notes[insert_ind][2]
                insert_ind += 1
            if not (insert_ind < len(notes) and notes[insert_ind][0] == note and notes[insert_ind][1] == 0):
                if insert_ind < len(notes):
                    notes[insert_ind][2] -= length
                notes.insert(insert_ind, [note, 0, length])
            del data[3]

        i += 1
    return notes


def convert(file):
    mid = mido.MidiFile(file)
    ticks_per_beat = mid.ticks_per_beat
    simple = []
    simple2 = []
    offset = 0

    for i, msg in enumerate(mido.merge_tracks(mid.tracks)):
        if msg.type[:4] == "note":
            note = msg.note
            vel = msg.velocity
            time = msg.time/ticks_per_beat + offset

            if vel != 0:
                simple2.append([note, vel, time-offset])
                simple.append([note, vel, time, 0])
                offset = 0

            else:
                offset = time
                ind = len(simple) - 1
                length = time
                # Loop through end of list until all note with current node value's length is set
                while ind >= 0:
                    if simple[ind][0] == note:
                        if simple[ind][3] == 0:
                            simple[ind][3] = length
                        else:
                            break

                    time = simple[ind][2]
                    length += time
                    ind -= 1

    data = [str(x) for x in simple]

    data = np.array(list(map(lambda x: ast.literal_eval(x), data)))

    with open(os.path.join("data", os.path.split(file)[-1][:-3]+"npy"), 'wb') as f:
        np.save(f, data)

    return data


def convert_all(directory, convert_func):
    directory = os.fsencode(directory)
    vocab = set()
    for file in os.listdir(directory):
        file_name = os.fsdecode(file)
        v = convert_func(os.path.join(os.fsdecode(directory), file_name))
        if v is not None:
            vocab = vocab.union(v)

    if vocab:
        return vocab


if __name__ == "__main__":
    conved = convert_to_matrix("midis/DEB_PASS.mid")
    make_midi_file(conved, "temp.mid")



