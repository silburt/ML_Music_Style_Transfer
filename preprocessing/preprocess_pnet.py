import numpy as np
import librosa
import soundfile as sf
from intervaltree import Interval, IntervalTree
from scipy import fft 
import pretty_midi
import pickle
import h5py
import sys
import argparse
import os
import glob
from utils.pretty_midi_roll_to_midi import piano_roll_to_pretty_midi

DEBUG_DIR = '/Users/arisilburt/Machine_Learning/ML_Music_Style_Transfer/preprocessing/debugdir'

class hyperparams(object):
    '''
    Definitions:
        - window: a pianoroll column
        - chunk: the entire pianoroll segment that will constitude a data point (X windows make a chunk)
    '''
    def __init__(self):
        self.sr = 44100 # Sampling rate (samples per second)
        self.n_fft = 2048 # fft points (samples)
        self.stride = 256 # number of windows/columns of separation between chunks

        self.piano_scores = [1760, 2308, 2490, 2491, 2527, 2533]
        self.styles = ['aliciakeys', 'cuba', 'gentleman', 'harpsichord', 'markisuitcase', 'upright']
        
        # A.S. each song is chopped into windows, and I *think* hop is the window length?
        self.ws = 256   # window size (audio samples per window)
        self.wps = 44100 // self.ws # ~172 windows/second
        self.spc = 5    # seconds per chunk

hp = hyperparams()

# mostly for debugging
def write_chunked_samples(audio_chunk, pianoroll_chunk, out_dir, style, song_id, step):
    outpath = os.path.join(out_dir, f"{song_id}_{style}_s{step}")
    sf.write(outpath + ".wav", audio_chunk, hp.sr)

    reformatted_pianoroll_chunk = pianoroll_chunk.T.astype('int') * 127
    midi_slice = piano_roll_to_pretty_midi(reformatted_pianoroll_chunk, fs=hp.wps)
    midi_slice.write(outpath + ".mid")


def write_h5py(train_data, spec_list, score_list, onoff_list, inst, index):
    '''
    Incrementally add to an h5py file, so that eveything can fit in memory
    '''
    # https://stackoverflow.com/questions/47072859/how-to-append-data-to-one-specific-dataset-in-a-hdf5-file-with-h5py
    if index == 0:
        print('creating datasets')
        train_data.create_dataset(inst + "_spec", data=spec_list, dtype='float32', maxshape=(None,) + spec_list.shape[1:], chunks=True) 
        train_data.create_dataset(inst + "_pianoroll", data=score_list, dtype='float64', maxshape=(None,) + score_list.shape[1:], chunks=True) 
        train_data.create_dataset(inst + "_onoff", data=onoff_list, dtype='float64', maxshape=(None,) + onoff_list.shape[1:], chunks=True) 
    else:
        print('appending to datasets')
        train_data[inst + "_spec"].resize(train_data[inst + "_spec"].shape[0] + spec_list.shape[0], axis=0)
        train_data[inst + "_spec"][-spec_list.shape[0]:] = spec_list

        train_data[inst + "_pianoroll"].resize(train_data[inst + "_pianoroll"].shape[0] + score_list.shape[0], axis=0)
        train_data[inst + "_pianoroll"][-score_list.shape[0]:] = score_list

        train_data[inst + "_onoff"].resize(train_data[inst + "_onoff"].shape[0] + onoff_list.shape[0], axis=0)
        train_data[inst + "_onoff"][-onoff_list.shape[0]:] = onoff_list


def process_spectrum_from_chunk(audio_chunk):
    spec = librosa.stft(audio_chunk, n_fft=hp.n_fft, hop_length=hp.stride)
    magnitude = np.log1p(np.abs(spec)**2)
    return magnitude


def split_chunk(audio, pianoroll, onoff, step, debug=False):
    '''
    Split audio into a windowed chunk
    '''
    n_samples = hp.spc * hp.sr
    audio_chunk = audio[(step * hp.ws * hp.stride): (step * hp.ws * hp.stride) + n_samples] 
    
    n_windows = hp.spc * hp.wps
    pianoroll_chunk = pianoroll[(step * hp.stride): (step * hp.stride) + n_windows]
    onoff_chunk = onoff[(step * hp.stride): (step * hp.stride) + n_windows]
    if debug is True:
        print('n_windows', n_windows)
        print('audio/midi shape', audio_chunk.shape, pianoroll_chunk.shape)
        print('audio/midi chunk percentages', len(audio_chunk)/len(audio), pianoroll_chunk.shape[0]/pianoroll.shape[0])
    return audio_chunk, pianoroll_chunk, onoff_chunk


def process_datum_into_chunks(audio, pianoroll, onoff, style, song_id, debug=True):
    '''
    Data Pre-processing
        
    Score: 
        Generate pianoroll from interval tree data structure
    
    Audio: 
        Convert waveform into power spectrogram

    '''
    spec_list=[]
    score_list=[]
    onoff_list=[]

    song_length = len(audio)
    num_chunks = (song_length) // (hp.wps * hp.stride)   # number chunks per song
    print('song has {} chunks'.format(num_chunks))

    for step in range(num_chunks - 30):   # A.S. why -30?
        if step % 50 == 0:
            print ('{} steps of song has been done'.format(step)) 
        audio_chunk, pianoroll_chunk, onoff_chunk = split_chunk(audio, pianoroll, onoff, step, debug=debug) 
        if debug == True:
            # check that the windowing alignment between midi/audio is correct
            write_chunked_samples(audio_chunk, pianoroll_chunk, DEBUG_DIR, style, song_id, step)

        spec_list.append(process_spectrum_from_chunk(audio_chunk))
        score_list.append(pianoroll_chunk)
        onoff_list.append(onoff_chunk)
    return np.array(spec_list), np.array(score_list), np.array(onoff_list)


def load_audio(data_dir, song_id, style, debug=False):
    audio_file = glob.glob(f"{data_dir}/{song_id}*{style}.wav")
    if len(audio_file) == 0:
        raise ValueError("couldnt find midi track!")

    y, sr = librosa.load(audio_file[0], sr=hp.sr)
    if debug is True:
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
        beat_times = librosa.frames_to_time(beat_frames, sr=sr)
        print("tempo: ", tempo)
        print("beat times", beat_times)
        print("beat frames:", beat_frames)
    return y


def load_midi(data_dir, song_id, ext='mixcraft'):
    midi_file = glob.glob(f"{data_dir}/{song_id}*{ext}.mid")
    if len(midi_file) == 0:
        raise ValueError("couldnt find midi track!")

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
    return pianoroll, onoff


def get_data(data_dir):
    '''
    Extract the desired solo data from the dataset.
    '''

    with h5py.File(os.path.join(data_dir, f'train_data_piano.hdf5'), 'a') as train_data:
        for song_id in hp.piano_scores:
            # get midi inputs
            pianoroll, onoff = load_midi(data_dir, song_id)
            
            for style in hp.styles:
                print(f"processing {style} style for song_id {song_id}")
                audio = load_audio(data_dir, song_id, style)

                spec_list, score_list, onoff_list = process_datum_into_chunks(audio, pianoroll, onoff, style, song_id)

                #write_h5py(train_data, spec_list, score_list, onoff_list, inst, index)


def main(args):
    get_data(args.data_dir)
   

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-data-dir", type=str, default='/Users/arisilburt/Machine_Learning/ML_Music_Style_Transfer/data/style_transfer_train', 
                        help="directory where dataset is is")
    args = parser.parse_args()
    
    main(args)
    
