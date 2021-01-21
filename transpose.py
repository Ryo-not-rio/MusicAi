import glob
import os
import music21

# converting everything into the key of C major or A minor

majors = dict(
    [("A-", 4), ("A", 3), ("B-", 2), ("B", 1), ("C", 0), ("C#", -1), ("D-", -1), ("D", -2), ("E-", -3), ("E", -4), ("F", -5),
     ("G-", 6), ("G", 5)])
minors = dict(
    [("A-", 1), ("A", 0), ("B-", -1), ("B", -2), ("C", -3), ("C#", -1), ("D-", -4), ("D", -5), ("E-", 6), ("E", 5), ("F", 4),
     ("G-", 3), ("G", 2)])

def transpose(directory):
    os.chdir("./"+directory)
    for file in glob.glob("*.mid"):
        score = music21.converter.parse(file)
        key = score.analyze('key')

        if key.tonic.name not in "CA":
            if key.mode == "major":
                halfSteps = majors[key.tonic.name]

            elif key.mode == "minor":
                halfSteps = minors[key.tonic.name]

            else:
                print("Error, key undefined")
                print(file)
                break

            newscore = score.transpose(halfSteps)
            # key = newscore.analyze('key')
            newFileName = file
            newscore.write('midi', newFileName)
