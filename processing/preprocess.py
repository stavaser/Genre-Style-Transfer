import os
import pypianoroll
import numpy as np
import glob
from song import Song
import librosa
import librosa.display
from postprocess import get_spectrogram
import pickle

class Preprocess():
    """Contains functions that preprocess a midi file into a .npy file.

    Attributes
    ----------
    file_path : str
        Root path.
    genres : list(str) -> dict(str: int)
        Dictionary of genres. Each genre must be subdirectory of root, e.g. `./root/<GENRE>`. Paired with
        their label in accordance with the current setting.
    bars : int
        Number of bars/segment.
    categories : list
        List of instrument categories according to the General MIDI (GM) standard. Excludes Percussive and
        Sound Effects. Read more at: https://www.midi.org/specifications-old/item/gm-level-1-sound-set.
    all_segments_directory : str
        Target path for processed files.
    resolution : int = 8
        Sets the resolution of the pianoroll (i.e. sampling rate).
    pitch_range : set = (0, 127)
        If user selects values, the read midi will have a reduced range. Values must be in range (0, 127).
    new_setting : bool = True
        Determines if the current session is running under a new setting. If the data is to be saved to the
        same directory, this attribute is to be set to 'False'. Setting it to 'True' will create a new
        setting and the processed data will be saved in a seperate directory, i.e. not the default
        all_segments_directory (./song_segments/) unless it has not been created yet.
    """
    def __init__(self, file_path: str,
                 genres: list,
                 bars : int,
                 resolution: int = 8,
                 pitch_range : set = (0, 127),
                 new_setting = True,
                 save_as_spectrogram = False):

        self.file_path = file_path
        self.dataset = {genre: [] for genre in genres}
        self.test_dataset = {genre: {} for genre in genres}
        self.genres = genres
        self.bars = bars
        self.resolution = resolution
        self.pitch_lower_bound = pitch_range[0]
        self.pitch_upper_bound = pitch_range[1]
        self.new_setting = new_setting
        self.categories = ['Piano', 'Chromatic Percussion', 'Organ',
                           'Guitar', 'Bass', 'Strings', 'Ensemble',
                           'Brass', 'Reed', 'Pipe', 'Synth Lead',
                           'Synth Pad', 'Synth Effects', 'Ethnic']

        self.spectrogram = save_as_spectrogram
        self.longest_spec = 0

        assert 0 <= self.pitch_lower_bound < 128, "Lower pitch bound out of range (0-127)"
        assert 0 <= self.pitch_upper_bound < 128, "Upper pitch bound out of range (0-127)"
        assert self.pitch_lower_bound < self.pitch_upper_bound, "Upper pitch bound must be higher than lower \
                                                                 pitch bound"

        self.all_segments_directory = './segments_inst1_bars{}_res{}_prange{}/'.format(bars, resolution, pitch_range)
        if self.new_setting:
            print("New setting requested...")
            iter_name = 0
            while(os.path.exists(self.all_segments_directory)):
                iter_name += 1
                self.all_segments_directory = './song_segments_' + str(iter_name) + '/'

        print("Processed music will be saved in:", self.all_segments_directory)

    def read_midi(self):
        """Reads all songs from directories specified by `self.genres`.

        Does not process songs already processed, and stored in ´self.all_segments_directory´.
        If the above directory does not exist, create it.
        Calls process_midi for all read midi files.
        """
        genre_songs = {}

        if not os.path.exists(self.all_segments_directory):
            os.mkdir(self.all_segments_directory)

        for idx, genre in enumerate(self.genres):
            if not os.path.exists(os.path.join(self.all_segments_directory, genre)):
                os.mkdir(os.path.join(self.all_segments_directory, genre))
            print("Loading files from:", genre)
            genre_path = os.path.join(self.file_path, genre)

            genre_songs[genre] = []
            for idx, songname in enumerate(os.listdir(genre_path)):
                # Make sure to only read '.mid' files
                if len(glob.glob(os.path.abspath(self.all_segments_directory).replace("\\", "/") + "/"
                 + songname.split('.mid')[0] + '*.npy')) == 0:
                    song = Song(genre_path, genre, songname)
                    genre_songs[genre].append(song)
                    #if song is not None:
                    #    if self.shortest_song_length == -1 or song.pmObject.get_end_time() < self.shortest_song_length:
                    #        self.shortest_song_length = 60 #song.pmObject.get_end_time()

        for genre in self.genres:
            genre_songs[genre] = list(filter(None, genre_songs[genre]))
            for song_idx, song in enumerate(genre_songs[genre]):
                print('{} ({}/{}) in {} \t\t'.format(song.name, str(song_idx + 1),
                                                        str(len(genre_songs)), genre), end='\r')
                try:
                    self.process_midi(song)
                except Exception as e:
                    print(e)

        if self.spectrogram:
            dataset = {genre: [] for genre in self.dataset.keys()}
            for genre in self.dataset.keys():
                for idx, spec in enumerate(self.dataset[genre]):
                    spec = (spec - np.amin(spec)) / (np.amax(spec) - np.amin(spec))
                    spec = np.pad(spec, ((0, 0), (0, self.longest_spec - spec.shape[1])), 'constant', constant_values=0.0)
                    dataset[genre].append(spec)

            file = open(os.path.join(self.all_segments_directory, 'dataset_spec.pkl'), 'wb')
            pickle.dump(dataset, file)
            file.close()

        else:
            file = open(os.path.join(self.all_segments_directory, 'dataset.pkl'), 'wb')
            pickle.dump(self.dataset, file)
            file.close()
            file2 = open(os.path.join(self.all_segments_directory, 'test_dataset.pkl'), 'wb')
            pickle.dump(self.test_dataset, file2)
            file2.close()

    def process_midi(self, song : Song):
        """Apply all preprocessing steps.

        Parameters
        ----------
        song : Song
            Object to preprocess.
        """
        if not self.has_valid_time_signature(song): return
        song = self.set_first_beat_to_zero(song)
        song = self.merge_similar_tracks(song)
        self.song_to_segments(song)

    def has_valid_time_signature(self, song : Song):
        """Validates time signature.

        The time signature of the song must be 4/4 and constant during the entire song.

        Parameters
        ----------
        song : Song
            Song that is to be validated.

        Returns
        -------
        Boolean
            True if valid time signature, else False.
        """
        sig_changes = song.pmObject.time_signature_changes
        if len(sig_changes) != 1:
            return False
        else:
            time_signature = '{}/{}'.format(sig_changes[0].numerator,
                                            sig_changes[0].denominator)
            if time_signature == '4/4':
                return True
            else:
                return False

    def set_first_beat_to_zero(self, song : Song):
        """Crops array to start at the bar closest to the first note.

        Parameters
        ----------
        song : Song
            Song that is to be trimmed.

        Returns
        -------
        Song
            Trimmed Song object.
        """
        estimated_start_time = song.pmObject.estimate_beat_start()
        beats = song.pmObject.get_downbeats()

        if estimated_start_time == 0.:
            return song

        # Start at the bar in which the estimated first beat is located.
        for beat in beats:
            if beat < estimated_start_time:
                real_start_time = beat
            else:
                break

        multitrack = pypianoroll.from_pretty_midi(song.pmObject, algorithm='custom',
                                                  first_beat_time=real_start_time, resolution=self.resolution)
        temp_song = pypianoroll.to_pretty_midi(multitrack)
        song.update_pianoroll(temp_song)
        return song

    def merge_similar_tracks(self, song : Song):
        """Merge similar instruments into the same PrettyMIDI.Instrument object.

        There are 128 instruments according to the MD 1 standard. This function merges instruments that belong
        to the same category of instruments. For example, there are 8 different guitars, all of which will be
        merged into the same track.

        The function will create a track for a category of instruments even if there are no instruments from
        that category in `song`. Tracks like this will be filled with zeros, and they allow the input shape of
        each song segment to be the same. This is needed because an eventual model will require the input of
        each datapoint to be the same.

        Parameters
        ----------
        song : Song
            The Song object that will have its tracks merged.

        Returns
        -------
        Song
            Song object with 14 tracks, one for each instrument category.
        """
        multitrack = pypianoroll.from_pretty_midi(song.pmObject, resolution=self.resolution)
        merged_track = pypianoroll.Multitrack(tempo=song.bpm(), name=song.name, resolution=self.resolution)
        sample_track_pianoroll = multitrack.tracks[0].pianoroll


        for category_idx in range(10): # Skip last two categories (Percussive and Sound Effects).
            category_tracks = pypianoroll.Multitrack(tempo=song.bpm(), resolution=self.resolution)
            for track in multitrack.tracks:
                if track.program // 8 == category_idx and not track.is_drum:
                    category_tracks.append(track)

            # If there are tracks from the category, merge them. If not a blank track will be created.
            try:
                blended_track = category_tracks.blend('max')
            except:
                blended_track = np.zeros_like(sample_track_pianoroll)

            track = pypianoroll.StandardTrack(name=self.categories[category_idx], program=category_idx*8,
                                              is_drum=False, pianoroll=blended_track)
            merged_track = merged_track.append(track)

        pianoroll = pypianoroll.to_pretty_midi(merged_track)
        song.update_pianoroll(pianoroll)
        return song

    def song_to_segments(self, song : Song):
        """Divide Song object into segments and save to .npy files.

        The input Song object will be divided into its maximum number of segments, based on `bars`. If the
        last segment is too short it will be discarded in order to keep all segments the same size. Will set
        the pitch range based on `self.lower_pitch_bound` and `self.upper_pitch_bound`.

        The segments will be saved to the directory `./song_segments/<GENRE/>`.

        Parameters
        ----------
        song : Song
            The Song object that will be turned into segments.
        """
        song_idx = 0

        self.test_dataset[song.genre][song.name] = []

        segment_length = self.bars * self.resolution * 4 # 4 is number of beats / bar

        multitrack = pypianoroll.from_pretty_midi(song.pmObject, resolution=self.resolution)

        while multitrack.get_length() > segment_length:
            segment = multitrack.copy()
            segment.trim(0, segment_length)
            try:
                song_array = segment.stack()
                song_array = self._reduce_pitch_range(song_array,
                                                      self.pitch_lower_bound,
                                                      self.pitch_upper_bound)

                if self.spectrogram:
                    pm_obj = self._segment_array_to_prettymidi(song_array, song.bpm(), song.name)
                    audio = pm_obj.synthesize(22050)
                    spectrogram = get_spectrogram(audio)

                    if spectrogram.shape[1] > self.longest_spec:
                        self.longest_spec = spectrogram.shape[1]

                    self.dataset[song.genre].append(spectrogram)

                else:
                    if np.sum(song_array) > 80 * 16:
                        song_array = np.amax(song_array, axis=0)
                        self.dataset[song.genre].append(song_array)
                        self.test_dataset[song.genre][song.name].append(song_array)

            except Exception as e:
                print(e)
                break

            multitrack.trim(start=segment_length)
            song_idx += 1

    def _segment_array_to_prettymidi(self, segment_array, bpm, name):
        multitrack = pypianoroll.Multitrack(tempo=np.array([120.0]), name=name, resolution=self.resolution)
        for idx, ch_roll in enumerate(segment_array):
            multitrack = multitrack.append(pypianoroll.StandardTrack(name=self.categories[idx], program=idx*8,
                                              is_drum=False, pianoroll=ch_roll))

        return pypianoroll.to_pretty_midi(multitrack, default_tempo=120.0)

    def _reduce_pitch_range(self, song : np.ndarray, lower_bound : int, upper_bound : int):
        """Limits the pitch range of the input song.

        Parameters
        ----------
        song : Song
            The Song object that will be clipped.
        lower_bound : int
            The lowest note number (pitch) to include.
        upper_bound : int
            The highest note number (pitch) to include.

        Returns
        -------
        song : Song
            Clipped Song object.
        """
        return song[:, :, lower_bound:upper_bound+1]
