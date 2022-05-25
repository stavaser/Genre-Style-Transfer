from preprocess import Preprocess

FILEPATH = "../Datasets/"
genres = ['Classic', 'Jazz']

def main():
    pp = Preprocess(file_path = FILEPATH,
                    genres = genres,
                    bars = 4,
                    resolution = 4,
                    pitch_range=(21,107),
                    new_setting=True,
                    save_as_spectrogram = False)
    pp.read_midi()

if __name__ == '__main__':
    main()