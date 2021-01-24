"""
Handling all the processing of the midi files
"""
import mido
import os
import ast
import numpy as np

BASE_TICKS_PER_BEAT = 256

def make_midi_file(unconverted_notes, file="new.mid"):
    if len(unconverted_notes[0]) == 4:
        unconvert_func = unconvert
    elif len(unconverted_notes[0]) == 127:
        unconvert_func = unconvert_matrix
    else:
        raise Exception("conversion protocol not recognised")

    notes = unconvert_func(unconverted_notes)
    new_mid = mido.MidiFile()
    new_mid.ticks_per_beat = BASE_TICKS_PER_BEAT
    new_track = mido.MidiTrack()
    new_mid.tracks.append(new_track)


    data = ["control_change channel=0 control=121 value=0 time=0",
            "program_change channel=0 program=0 time=0",
            "control_change channel=0 control=7 value=100 time=0",
            "control_change channel=0 control=10 value=64 time=0",
            "control_change channel=0 control=91 value=0 time=0",
            "control_change channel=0 control=93 value=0 time=0",
            "<meta message midi_port port=0 time=0>"] # Meta messages at the beginning
    # new_track.append(mido.MetaMessage('time_signature', numerator=4, denominator=4, clocks_per_click=24,
    #                                   notated_32nd_notes_per_beat=8, time=0))
    # new_track.append(mido.MetaMessage('key_signature', key='C', time=0))
    # new_track.append(mido.MetaMessage('set_tempo', tempo=500000, time=0))

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


def convert_to_matrix(file):
    sequence = []
    mid = mido.MidiFile(file)
    ticks_per_beat = mid.ticks_per_beat
    mult = BASE_TICKS_PER_BEAT/ticks_per_beat

    start = False
    step_matrix = [0]*127
    show = []

    for i, msg in enumerate(mido.merge_tracks(mid.tracks)):
        if msg.type[:4] == "note":
            note = msg.note
            vel = msg.velocity
            time = round(msg.time * mult)

            show.append([note, vel, time])

            if time > 0 and start:
                sequence.append(step_matrix[:])
                step_matrix = [0 if x == 0 else 1 for x in step_matrix]
                sequence += [step_matrix[:]]*(time-1)

            if vel != 0:
                step_matrix[note] = 2
            else:
                step_matrix[note] = 0

            if not start: start = True

    sequence.append(step_matrix)

    sequence = np.array(sequence)

    with open(os.path.join("data2", os.path.split(file)[-1][:-3] + "npy"), 'wb') as f:
        np.save(f, sequence)

    return set(sequence.flatten())


def unconvert_matrix(sequence_list):
    notes = []
    playing = [0]*127
    time = 0
    for matrix in sequence_list:
        for i, v in enumerate(matrix):
            count = 0
            if v == 2:
                notes.append([i, 80, time])
                playing[i] += 1
                time = 0
                count += 1

            elif v == 0 and playing[i]:
                notes.append([i, 0, time])
                playing[i] -= 1
                time = 0
                count += 1

        time += 1
    return notes


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



