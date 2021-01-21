import requests
from urllib.parse import urlparse, urljoin
from urllib.request import urlretrieve
from bs4 import BeautifulSoup
import re
import mido
from functools import reduce
import os
import shutil

from preprocess import process

urls = set()

def is_valid(url):
    parsed = urlparse(url)
    return bool(parsed.netloc) and bool(parsed.scheme)


def get_all_links(url):
    urls = set()
    soup = BeautifulSoup(requests.get(url).content, "html.parser")

    for a_tag in soup.findAll("a"):
        href = a_tag.attrs.get("href")
        if href == "" or href is None:
            continue

        href = urlparse(urljoin(url, href))
        href = href.scheme + "://" + href.netloc + href.path

        if not is_valid(href) or not re.search("mid$", href):
            continue

        urls.add(href)

    return urls


def get_midis_in_directory(directory):
    midis = []
    for file in os.listdir(directory):
        file_name = os.fsdecode(file)
        midis.append(mido.MidiFile("midis/" + file_name))
    return midis

def get_first_notes(midi, length=200):
    track = mido.merge_tracks(midi.tracks)
    notes = []
    count = 0
    ind = 0
    while count < length:
        msg = track[ind]

        if msg.type[:4] != "note":
            ind += 1
            continue

        notes.append([msg.note, msg.velocity, msg.time])
        count += 1

    return notes


def save_valid_midis(url, directory="midis"):
    valid_instruments = ["piano", "harpsichord"]
    links = get_all_links(url)
    prev_midis = [get_first_notes(x) for x in get_midis_in_directory("midis")]

    os.mkdir("temp")
    for link in list(links):
        name = re.findall("/[^/]+", link)[-1][1:]
        urlretrieve(link, "temp.mid")

        midi = mido.MidiFile("temp.mid")

        if len(midi.tracks) > 3:
            continue

        valid = True
        for track in midi.tracks[1:]:
            valid = valid and reduce(lambda a, b: a in track.name.lower() or b in track.name.lower(), valid_instruments)

        if not valid:
            continue

        notes = get_first_notes(midi)
        for notes1 in prev_midis:
            if check_two_midis_similar(notes1, notes):
                valid = False
                break

        if not valid:
            continue

        prev_midis.append(notes)

        midi.save(filename="temp" + "/" + name)

    process("temp")

    for file in os.listdir("temp"):
        shutil.copy("temp/" + file, "midis/" + file)

    shutil.rmtree("temp")
    print("downloaded all")


def check_two_midis_similar(notes1, notes2, boundary=0.6):
    same = 0
    for i, note1 in enumerate(notes1):
        if note1 == notes2[i]:
            same += 1

    return same/len(notes1) >= boundary


if __name__ == '__main__':
    URL = "https://www.mfiles.co.uk/classical-midi.htm"
    save_valid_midis(URL)
    # print(check_two_midis_similar(mido.MidiFile("temp.mid"), mido.MidiFile("temp.mid")))