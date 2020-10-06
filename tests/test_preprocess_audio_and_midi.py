# This tests the slicing of the audio/midi to make sure alignment is working properly and also that midi is being converted properly

import os
import sys
import librosa
sys.path.append('../preprocessing')
sys.path.append('../preprocessing/utils')
from preprocess_audio_and_midi import load_audio
from midi_to_vec import midi_file_to_array, array_to_midi_file


def test_slicing(midi_file, audio_file, basename, output_dir, slice_factor):
    # write names
    midi_outpath = os.path.join(output_dir, f'{basename}_slice{slice_factor}.mid')
    audio_outpath = os.path.join(output_dir, f'{basename}_slice{slice_factor}.wav')
    
    # load, slice, write audio
    audio, sr, tempo = load_audio(audio_file)
    audio_index = int(len(audio) / slice_factor)
    audio_slice = audio[:audio_index]
    librosa.output.write_wav(audio_outpath, audio_slice, sr)
    
    # load, slice, write midi
    piano_roll = midi_file_to_array(midi_file)
    midi_index = int(len(piano_roll) / slice_factor)
    piano_roll_slice = piano_roll[:midi_index, :]
    array_to_midi_file(piano_roll_slice, midi_outpath, tempo_bpm=tempo)


if __name__ == '__main__':
    midi_file = '../data/musicnet_midis/Bach/2303_prelude5.mid'
    audio_file =  '../data/musicnet/test_data/2303.wav'
    basename = '2303'
    output_dir = 'test_outputs'
    
    slice_factor = 5    # divide the files by `slice_factor` and write to file
    
    test_slicing(midi_file, audio_file, basename, output_dir, slice_factor)
    
