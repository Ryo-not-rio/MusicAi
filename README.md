# Music Generating Ai
This is an attempt to use neural networks to generate music from scratch. Tensorflow is used for the neural network process and a module called mido is used to process music data. This is further explained in later sections.

## Table of contents
 - [Data used](#data-used)
 - [Approach 1](#approach-1)
	 - [Approach 1.1](#approach-1.1)
 - [Midi file format](#midi-format)


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

## Approach 1.1
The first approach is to treat the problem as same as a text generating problem. In text generating problems, a recurrent neural network (RNN) is used to try to predict the next character of a text input. The RNN is fed a series of texts, looks at each word, try to predict what the next letter should be and weights are adjusted according how low the model's probability on choosing the right letter was. More detail about this approach and how to implement it is found [here](https://www.tensorflow.org/tutorials/text/text_generation). 

To form the data into a compatible format for this approach, not much work is needed as the midi file is already a sequence. The only formatting that was done was to read all the notes in the midi files and convert it into arrays of format :
[[note, velocity, time], [note, velocity, time] ...]

In case of text generation, the output of the AI model is an array of probabilities where the index corresponds to a word within the vocabulary fed to the AI i.e. the output shape is (vocab_size). However, in case of music, the AI needs to predict the note, velocity and time so the output needs to be separated into three arrays, each corresponding to the probability distribution for note, velocity and time respectively i.e. output shape is (note_vocab_size, velocity_vocab_size, time_vocab_size). This can be done by splitting the model at some point into three separate networks and the point at which to split the network can be experimented with as long as the output shape remains correct.

With this model, the most optimal accuracy achieved was about 0.5, 0.3, 0.6 for note, velocity and time respectively. The music generated by this model is frankly not music, and is characterized by very long notes and a lot of dissonance. The long notes are definitely caused by the model failing to predict note off events for notes that are playing, and the amount of wrong notes may indicate the model guessing the note for a note off event but predicting a positive velocity.

The following problems were identified as potential causes of the bad performance.

1. In music, a lot of what note is played is influenced by what note was played before and what note is currently playing. To figure out what note is currently being played, first, the model has to learn that a velocity of 0 corresponds to a note being turned off, and then comb through all the previous data and see for each note how many note off events there were after a note on event. 
2. The sequence provided gives no distinction between a note on and note off event except for the velocity. This means the model treats guessing a velocity of 80 where it should have been 60 the same way as guessing a velocity of 20 where it should have been 0.
3. The model relies heavily on all three attributes being correct. For example, the model may predict a note off event at the correct time but if the note predicted is even off by one, the model turns off a wrong note or the prediction has no effect if that note is not currently being played.

## Attempt 1.2
This approach aims to address the issue of the model treating note on and off events as the same and help figure out what notes are currently being played.
