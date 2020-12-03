"""
Handling all the processing of the midi files
"""
import mido
import os

# Make a midi file given a list of notes
def make_midi_file(notes):
    notes = unconvert(notes) # Unformat formatted list of notes
    new_mid = mido.MidiFile()
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

    new_mid.save('new.mid')
    print("successfully saved midi file")


# Unformatting each note to the format [note, velocity, time]
def unconvert(notes):
    i = 0
    while i < len(notes):
        data = notes[i]
        note, vel, time = data[0], data[1], data[2]
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


# Converting all data midi files
# from [note, velocity, time] to [note, velocity, time, length]
# Also removing all notes with a velocity of 0.
# The result is written to the folder data
def convert(directory="midis"):
    directory = os.fsencode(directory)

    for file in os.listdir(directory):
        file_name = os.fsdecode(file)

        mid = mido.MidiFile("midis/" + file_name)
        simple = []
        simple2 = []
        offset = 0
        for i, msg in enumerate(mido.merge_tracks(mid.tracks)):
            msg = str(msg).split(" ")
            if msg[0] == "note_on":
                note = int(msg[2].split("=")[1])
                vel = int(msg[3].split("=")[1])
                time = int(msg[4].split("=")[1]) + offset
                simple2.append([note, vel, time-offset])

                if vel != 0:
                    simple.append([note, vel, time, 0])
                    offset = 0
                else:
                    offset = time
                    ind = len(simple)-1
                    length = time
                    while ind >= 0:
                        if simple[ind][0] == note:
                            if simple[ind][3] == 0:
                                simple[ind][3] = length
                            else:
                                break

                        time = simple[ind][2]
                        length += time
                        ind -= 1

        with open(os.path.join("data", file_name[:-3]+"txt"), "w") as f:
            f.write(";".join([str(x) for x in simple]) + ";")
    print("Successfully converted all")


if __name__ == "__main__":
    convert()


