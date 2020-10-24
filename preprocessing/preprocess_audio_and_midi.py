# This is the general script that takes audio/midi files and prepares them as model inputs
# TODO: some slight alignment issues between audio/midi
# TODO: Look into https://pytorch.org/tutorials/beginner/audio_preprocessing_tutorial.html

from mido import MidiFile
import glob2
import os
from utils.midi_to_vec import midi_file_to_array
import librosa


def get_wavenet_midi_file(wav_midi_dir, csv_file):
    # get midi file
    name = csv_file.split('.csv')[0]
    pathdir = os.path.join(wav_midi_dir, 'musicnet_midis')
    midi_file = glob2.glob(f'{pathdir}/**/*{name}*.mid')
    if len(midi_file) > 1:
        raise ValueError(f'WARNING: retrieved more than one file: {midi_file}')
    midi_file = midi_file[0]
    return midi_file
    

def get_wavenet_audio_file(wav_midi_dir, csv_file, data_type):
    name = csv_file.split('.csv')[0]
    audio_file = os.path.join(wav_midi_dir, 'musicnet', f'{data_type}_data', f'{name}.wav')
    return audio_file


def load_audio(audio_file, debug=False):
    y, sr = librosa.load(audio_file)
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    if debug is True:
        print("tempo: ", tempo)
        print("beat times", beat_times)
        print("beat frames:", beat_frames)
    return y, sr, tempo


def main(data_list_path, wav_midi_dir, data_type, debug=False):
    with open(data_list_path) as f:
        lines = f.read().splitlines()
    
    for csv_file in lines:
        # load audio
        audio_file = get_wavenet_audio_file(wav_midi_dir, csv_file, data_type)
        audio, sr, tempo = load_audio(audio_file, debug=debug)
    
        # load midi (as an array)
        midi_file = get_wavenet_midi_file(wav_midi_dir, csv_file)
        piano_roll = midi_file_to_array(midi_file)
            
        break


if __name__ == '__main__':
    data_type = 'test'  # test or train
    data_list_path = f'piano_pieces/piano_pieces_{data_type}.txt'
    wav_midi_dir = '../data/'
    debug = False
    
    main(data_list_path, wav_midi_dir, data_type, debug)
    
    
