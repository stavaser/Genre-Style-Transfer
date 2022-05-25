import pretty_midi as pm
import numpy as np
import os

class Song():
    """Additional functionality from PrettyMIDI object.

    Reads midi (.mid) file and turns it into a PrettyMIDI object. Also extends functionality.

    Attributes
    ----------
    pmObject : PrettyMIDI
        Created from midi file based on `file_path` and `name`, or set to creates empty object if unable to read the file.
    file_path : str
        Path to specific genre folder.
    genre : int
        Genre index.
    name : str
        Name of the song.
    """
    def __init__(self, file_path: str, genre: str, filename: str):
        print("Loading song:", filename, "\t\t\t\t\t\t\t", end="\r")
        try:
            self.pmObject = pm.PrettyMIDI(file_path + '/' + filename)
        except Exception as e:
            print(e)
            self.pmObject = pm.PrettyMIDI()
        self.file_path = file_path
        self.genre = genre
        self.name = filename.split(".mid")[0]

    def bpm(self):
        """Gets the bpm for the song.

        If the song has several tempi, the first one will be returned.

        Returns
        -------
        Float
            The first tempo value.
        """
        _, tempi = self.pmObject.get_tempo_changes()
        tempo = np.array([tempi[0]])
        return tempo

    def update_pianoroll(self, new_pmObject):
        """Overrides the current value of pmObject with the input.

        Since PrettyMIDI doesn't have an `update_pianoroll` function, this appears to be the simplest solution.

        Parameters
        ----------
        new_pmObject : PrettyMIDI
            New PrettyMIDI object.
        """
        self.pmObject = new_pmObject
        return self
