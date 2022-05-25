import numpy as np
import pypianoroll
import librosa
import pretty_midi
import torch
import sys
sys.path.insert(1, '../')
import torch.nn.functional as F
import os

VALID_TYPES = {'original', 'cycle', 'generated'}

def npy2mid(song, gen, name, directory):
    categories = ['Piano', 'Chromatic Percussion', 'Organ',
                  'Guitar', 'Bass', 'Strings', 'Ensemble',
                  'Brass', 'Reed', 'Pipe', 'Synth Lead',
                  'Synth Pad', 'Synth Effects', 'Ethnic']

    song = pad_song(song)

    merged_song = pypianoroll.Multitrack(tempo=np.array([120.0], dtype=float), resolution=4)
    for category_idx, track in enumerate(song):
        temp_track = pypianoroll.StandardTrack(name=categories[category_idx],
                                               program=category_idx*8,
                                               is_drum=False,
                                               pianoroll=track)
        merged_song.append(temp_track)

    midi_song = pypianoroll.to_pretty_midi(merged_song)

    gen = pad_song(gen)

    merged_gen = pypianoroll.Multitrack(tempo=np.array([120.0], dtype=float), resolution=4)
    for category_idx, track in enumerate(gen):
        temp_track = pypianoroll.StandardTrack(name=categories[category_idx],
                                               program=category_idx*8,
                                               is_drum=False,
                                               pianoroll=track)
        merged_gen.append(temp_track)

    midi_gen = pypianoroll.to_pretty_midi(merged_gen)

    midi_song.write(directory + name + '_orig' +'.mid')
    midi_gen.write(directory + name + '_gen' +'.mid')

def npy_to_midi(song : torch.Tensor,
                index : int,
                song_type: str,
                directory : str,
                bpm : float = 120.0,
                resolution : int = 8):
    pass
    """Convert numpy array to midi format.

    The file is saved as `INDEX_SONGTYPE.midi`. An example would be `0_original.midi`. Input song will be
    padded using pad_song() to have the correct output size.

    Parameters
    ----------
    song : Tensor
        Song that should be saved to midi.
    index : int
        Used in filename.
    song_type : str
        Should be one of: [original, cycle, generated].
    directory : str
        The directory where the midi should be saved to.
    bpm : float
        Bpm (beats per minute) of the midi file.
    resolution : int
        Sets the resolution of the pianoroll (i.e. sampling rate). It is recommended to use the same
        resolution as used with preprocessing.
    """

    categories = ['Piano', 'Chromatic Percussion', 'Organ',
                  'Guitar', 'Bass', 'Strings', 'Ensemble',
                  'Brass', 'Reed', 'Pipe', 'Synth Lead',
                  'Synth Pad', 'Synth Effects', 'Ethnic']

    if song_type not in VALID_TYPES:
        raise ValueError("song_type must be one of %r" %VALID_TYPES)

    if song.dim() == 4:
        song = song[0]

    if song.is_cuda: # Make sure song Tensor is on cpu
        song = song.cpu()

    song = pad_song(song)
    merged_track = pypianoroll.Multitrack(tempo=np.array([bpm], dtype=float), resolution=resolution)
    for category_idx, track in enumerate(song):
        temp_track = pypianoroll.StandardTrack(name=categories[category_idx],
                                               program=category_idx*8,
                                               is_drum=False,
                                               pianoroll=track)
        merged_track.append(temp_track)
    midi_song = pypianoroll.to_pretty_midi(merged_track)

    if not os.path.exists(directory):
        os.mkdir(directory)

    midi_song.write('./' + directory + '/' + str(index) + '_' + song_type + '.mid')




def pad_song(song : torch.Tensor):
    """Adds padding to pytorch Tensor.

    Pads input Tensor to have the shape [<LENGTH>, 128]. Midi files need to have height 128 because thats the
    number of possible notes. Having a lower amount will not save the file correctly and notes might get
    shifted.

    Parameters
    ----------
    song : Tensor
        Tensor that should be padded.

    Returns
    -------
    track : Tensor
        Padded Tensor.
    """
    note_number_count = len(song[0][0])
    padding_size = (128 - note_number_count)//2
    track = F.pad(song, (padding_size, padding_size), 'constant', 0)
    return track

def get_spectrogram(audio):
    fmin = librosa.midi_to_hz(21)

    spec = librosa.cqt(audio, sr=22050, fmin=fmin, n_bins=87, bins_per_octave=12)
    spec = librosa.amplitude_to_db(np.abs(spec), ref=np.max)

    return spec