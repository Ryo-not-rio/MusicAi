# Music Generating Ai
This is an attempt to use neural networks to generate music from scratch. Tensorflow is used for the neural network process and a module called mido is used to process music data. This is further explained in later sections.

## Table of contents
 - [Data used](##data_used)
 - [Midi file format](##Midi_format)
 - [Approach 1](#approach_1)
	 - [Approach 1.1](##approach_1.1)
 


## Data used
The midi file format is used as the dataset. Midi files are used primarily to encode music for music making making software and electronic instruments. It is an ideal format for this use case as music is denoted as a sequence of events such as note on and note off. Furthermore, the meta data for each file contains what instruments are used and how many tracks are in the file which can be used to narrow down the style of music.

## Midi format
The midi file format is slightly counter intuitive so I have included a brief summary as understanding the format is crucial to understanding how the dataset is created and the architectures of the neural networks.

As stated in the previous section, the midi file format is structured as a series of events.
For this project, the important events are the note_on and note_off events and they are structured as follows:
[Event type, Note,  Velocity, Time]

**Event type**
The event type simply denotes if the note turned on or off. The note_on event signifies the start of a note, and the note_off event signifies the end of the note. This attribute is ignored as a note_on event with 0 velocity(loudness) is identical to a note_off event and depending on what software is used to create the midi_file, the note_off event may not be used entirely.

**Note**
This is an integer value from 0 to 127 inclusive that indicates what note the event is for. 60 corresponds to middle C and the increment/decrement and 1 note unit corresponds to half step in the conventional western music notation.

**Velocity**
This is an integer value from 0 to 127 inclusive that dictates the loudness of each note. As mentioned previously, a value of 0 dictates a note turning off.

**Time**
In midi files, the time unit is ticks per beat instead of beats per minute which is defined in the meta information of the file. It is this unit which is used to dictate when an event occurs. The ticks per beat simply denotes how many ticks make up one beat, so for example at 256 ticks per beat, a quarter note is 256 beats, an eighth note is 128 ticks and so on. This allows the sequence to be unaffected by the tempo of the music, however, each piece needs to be normalised to one standard ticks per beat.

The time attribute in events dictates how many ticks to wait after the previous event before firing the event. For example the event [note_on, 60, 100, 128] can be translated as "wait 128 ticks and then start playing a note of value 60(middle C) at a velocity of 100".

**Music Sequence**
To dictate a whole music file, these events are listed in order.
Below are some simplified examples at 256 ticks per beat, showing how a whole music sequence is denoted.

Example 1:
(beginning_of_file, [note_on, 60, 100, 128], [note_off, 60, 0, 256])
Translation: Wait 128 ticks, start playing note 60 at 100 velocity. Stop playing note 60 after 256 ticks.
In music terms: Wait half a beat, play middle C quarter note.

Example 2:
(beginning_of_file, [note_on, 60, 100, 128], [note_on, 64, 100, 0], [note_on, 67, 100, 0], [note_off, 60, 0, 512], [note_on, 64, 0, 0], [note_on, 67, 0, 0])
In this example, some of the time attributes are 0. This simply denotes fire this event simultaneously with the previous event.
Translation: Wait 128 ticks, start playing notes 60, 64 and 67 all at a velocity of 100. Wait 512 ticks then stop playing notes 60, 64, 67.
In music terms: Wait half a beat, play a C major chord for 2 beats.

# Approach 1

## Approach 1.1 (Naivate)

