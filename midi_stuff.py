import mido
import os


def encode(notes):
    return [x[0] * 1000000 + x[1] * 1000 + x[2] for x in notes]  # formula: note*10000 + vel*100 + time


def decode(encoded):
    for j, n in enumerate(encoded):
        note_val, remainder = divmod(n, 1000000)
        vel_val, time_val = divmod(remainder, 1000)
        encoded[j] = [note_val, vel_val, time_val]
    return encoded


def make_midi_file(notes):
    new_mid = mido.MidiFile()
    new_track = mido.MidiTrack()
    new_mid.tracks.append(new_track)

    data = ["control_change channel=0 control=121 value=0 time=0",
            "program_change channel=0 program=0 time=0",
            "control_change channel=0 control=7 value=100 time=0",
            "control_change channel=0 control=10 value=64 time=0",
            "control_change channel=0 control=91 value=0 time=0",
            "control_change channel=0 control=93 value=0 time=0",
            "<meta message midi_port port=0 time=0>"]
    # new_track.append(mido.MetaMessage('time_signature', numerator=4, denominator=4, clocks_per_click=24,
    #                                   notated_32nd_notes_per_beat=8, time=0))
    # new_track.append(mido.MetaMessage('key_signature', key='C', time=0))
    new_track.append(mido.MetaMessage('set_tempo', tempo=500000, time=0))

    for n in notes:
        try:
            new_track.append(mido.Message('note_on', note=n[0], velocity=n[1], time=n[2]))
        except ValueError as e:
            print(e)
            print(n)

    new_track.append(mido.MetaMessage('end_of_track', time=1))

    new_mid.save('new.mid')
    print("successfully saved midi file")


def convert(directory="midis"):
    directory = os.fsencode(directory)

    for file in os.listdir(directory):
        file_name = os.fsdecode(file)

        mid = mido.MidiFile("midis/" + file_name)
        simple = []
        for i, track in enumerate(mid.tracks):
            # print('Track {}: {}'.format(i, track.name))
            for msg in track:
                # print(msg)
                msg = str(msg).split(" ")
                if msg[0] == "note_on":
                    note = int(msg[2].split("=")[1])
                    vel = int(msg[3].split("=")[1])
                    time = int(msg[4].split("=")[1])

                    simple.append([note, vel, time])

        with open("data.txt", "a") as f:
            f.write(";".join([str(x) for x in simple]) + ";")


