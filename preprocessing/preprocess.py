import numpy as np
import librosa
from scipy import fft 
import pretty_midi
import pickle
import h5py
import sys
import argparse
import os
import glob
from utils import io_manager

ROOT_DIR = '/Users/arisilburt/Machine_Learning/ML_Music_Style_Transfer/'
DEBUG_DIR = f'{ROOT_DIR}/preprocessing/debugdir'

class hyperparams(object):
    '''
    Definitions:
        - window: a pianoroll column / unit of time
        - chunk: the entire pianoroll segment that will constitude a data point (X windows make a chunk)
    '''
    def __init__(self):
        self.sr = 44100 # Sampling rate (samples per second)
        self.n_fft = 2048 # fft points (samples)
        self.stride = 512 # number of windows of separation between chunks/data points

        self.piano_scores = {
            'train': [1760, 2308, 2490, 2527],  # 2491 errors out, not sure why
            'test': [2533]
        }
        self.styles = ['cuba', 'aliciakeys', 'gentleman', 'harpsichord', 'markisuitcase', 'upright']
        
        # A.S. each song is chopped into windows, and I *think* hop is the window length?
        self.ws = 256   # window size (audio samples per window)
        self.wps = 44100 // self.ws # ~172 windows/second
        self.spc = 5    # seconds per chunk

hp = hyperparams()


def process_spectrum_from_chunk(audio_chunk):
    spec = librosa.stft(audio_chunk, n_fft=hp.n_fft, hop_length=hp.stride)
    magnitude = np.log1p(np.abs(spec)**2)
    return magnitude


def process_audio_into_chunks(audio, style, song_id, num_chunks, debug=False):
    print(f"processing {style} style for song_id {song_id}")
    spec_list=[]
    for step in range(num_chunks):
        # get audio chunk
        n_samples_per_chunk = hp.spc * hp.sr
        audio_chunk = audio[(step * hp.ws * hp.stride): (step * hp.ws * hp.stride) + n_samples_per_chunk] 
        
        # check that the windowing alignment between midi/audio is correct
        if debug == True:
            io_manager.write_chunked_samples(DEBUG_DIR, song_id, step, hp, style=style, audio_chunk=audio_chunk)
        
        # append to lists
        spec_list.append(process_spectrum_from_chunk(audio_chunk))
    return np.array(spec_list)


def process_pianoroll_into_chunks(pianoroll, onoff, song_id, num_chunks, debug=False):
    print(f"processing pianoroll for song_id {song_id}")
    score_list=[]
    onoff_list=[]
    for step in range(num_chunks):
        # get pianoroll/onoff chunk
        n_windows_per_chunk = hp.spc * hp.wps
        pianoroll_chunk = pianoroll[(step * hp.stride): (step * hp.stride) + n_windows_per_chunk]
        onoff_chunk = onoff[(step * hp.stride): (step * hp.stride) + n_windows_per_chunk]
        if debug == True:
            # check that the windowing alignment between midi/audio is correct
            io_manager.write_chunked_samples(DEBUG_DIR, song_id, step, hp, pianoroll_chunk=pianoroll_chunk)

        # append
        score_list.append(pianoroll_chunk)
        onoff_list.append(onoff_chunk)
    return np.array(score_list), np.array(onoff_list)


def load_audio(data_dir, song_id, style, debug=False):
    audio_file = glob.glob(f"{data_dir}/{song_id}*{style}.wav")
    if len(audio_file) == 0:
        raise ValueError("couldnt find audio track!")
    elif len(audio_file) > 1:
        raise ValueError("multiple files picked up, issue:", audio_file)

    y, sr = librosa.load(audio_file[0], sr=hp.sr)
    if debug is True:
        #tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
        #beat_times = librosa.frames_to_time(beat_frames, sr=sr)
        #print("tempo: ", tempo)
        #print("beat times", beat_times)
        #print("beat frames:", beat_frames)
        print("length of audio clip / sr: ", len(y), sr)
        print("audio files picked up:", audio_file)
    return y


def get_num_song_chunks(pianoroll, offset_percentage=0.1):
    '''
    Get number of chunks in the song. 
    - offset_percentage is to allow a bit of wiggle room to make sure we dont accidentally run off the end. Some of the audio/midi
    lengths end at slightly different spots (bug to be fixed...)
    '''
    n_windows_per_chunk = hp.spc * hp.wps
    num_chunks = (pianoroll.shape[0] - n_windows_per_chunk) // hp.stride
    # audio way to calculate number of chunks
    #n_samples_per_chunk = hp.spc * hp.sr
    #num_chunks = (len(audio) - n_samples_per_chunk) // (hp.ws * hp.stride)
    
    offset = int(offset_percentage * num_chunks)
    num_chunks -= offset
    print('song has {} chunks'.format(num_chunks))
    return num_chunks


def load_midi(data_dir, song_id, ext='mixcraft', debug=False):
    midi_file = glob.glob(f"{data_dir}/{song_id}*{ext}.mid")
    if len(midi_file) == 0:
        raise ValueError("couldnt find midi track!")
    elif len(midi_file) > 1:
        raise ValueError("multiple files picked up, issue:", midi_file)

    midi = pretty_midi.PrettyMIDI(midi_file[0])    
    pianoroll = midi.get_piano_roll(fs=hp.wps).T
    pianoroll[pianoroll.nonzero()] = 1
    onoff = np.zeros(pianoroll.shape) 
    for i in range(pianoroll.shape[0]):
        if i == 0:
            onoff[i][pianoroll[i].nonzero()] = 1
        else:
            onoff[i][np.setdiff1d(pianoroll[i-1].nonzero(), pianoroll[i].nonzero())] = -1
            onoff[i][np.setdiff1d(pianoroll[i].nonzero(), pianoroll[i-1].nonzero())] = 1 
    
    if debug is True:
        print("length of pianoroll: ", pianoroll.shape)
        print("midi files picked up:", midi_file)
    return pianoroll, onoff


def get_data(data_dir, dataset_path_basename, data_type, debug=False):
    '''
    Extract the desired solo data from the dataset.
    '''
    
    h5pyname = f"{dataset_path_basename}_{data_type}.hdf5"
    with h5py.File(h5pyname, 'w') as h5py_data:
        data_manager = io_manager.h5pyManager(h5py_data)

        for song_id in hp.piano_scores[data_type]:
            # load midi
            pianoroll, onoff = load_midi(data_dir, song_id, debug=debug)
            num_chunks = get_num_song_chunks(pianoroll)

            # process into chunks
            pianoroll_list, onoff_list = process_pianoroll_into_chunks(pianoroll, onoff, song_id, num_chunks, debug=debug)
            
            # write
            data_manager.write_pianoroll(pianoroll_list, onoff_list)

            for style in hp.styles:
                # load audio
                audio = load_audio(data_dir, song_id, style, debug=debug)

                # process into chunks
                spec_list = process_audio_into_chunks(audio, style, song_id, num_chunks, debug=debug)
                
                # write
                data_manager.write_spectrum(spec_list, style)

                if debug is True:
                    assert pianoroll_list.shape[0] == spec_list.shape[0]
                    assert pianoroll_list.shape == onoff_list.shape


def main(args):
    get_data(args.data_dir, args.dataset_path_basename, args.data_type, args.debug)
   

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-data-dir", type=str, default=f'{ROOT_DIR}/data/style_transfer_train', 
                        help="directory where dataset is is")
    parser.add_argument("-dataset-path-basename", type=str, default=f'{ROOT_DIR}/preprocessing/data_products/style_transfer', 
                        help="directory where dataset is is")
    parser.add_argument("-data-type", type=str, default='train', choices=['train', 'test'],
                        help="directory where dataset is is")                                         
    parser.add_argument("--debug", type=io_manager.str2bool, default=False, 
                        help="whether to run in debug mode or not")                    
    args = parser.parse_args()
    
    main(args)
    
