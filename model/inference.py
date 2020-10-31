import argparse
import torch
import pretty_midi
import numpy as np
import h5py
import pickle
import torch.nn as nn
import torch.utils.data as utils
import json
import os
from model import PerformanceNet
import librosa
import soundfile as sf
from tqdm import tqdm
import sys
sys.path.append('../preprocessing/')
from preprocess import process_spectrum_from_chunk, hyperparams

preprocess_hp = hyperparams()


class AudioSynthesizer():
    def __init__(self, checkpoint, exp_dir, midi_source, audio_source):
        self.exp_dir = exp_dir
        self.checkpoint = torch.load(os.path.join(exp_dir, checkpoint))
        self.sample_rate = preprocess_hp.sr
        self.wps = preprocess_hp.wps
        self.midi_source = midi_source
        self.audio_source = audio_source
                
    def get_test_midi(self):
        X = np.load(os.path.join(self.exp_dir,'test_data/test_X.npy'))
        rand = np.random.randint(len(X),size=5)
        score = [X[i] for i in rand]
        return torch.Tensor(score).cuda()

    def process_custom_midi_and_audio(self, midi_filename, audio_filename):
        # process midi
        midi_dir = os.path.join(self.exp_dir, 'midi')
        midi = pretty_midi.PrettyMIDI(os.path.join(midi_dir, midi_filename))    
        pianoroll = midi.get_piano_roll(fs=self.wps).T
        pianoroll[pianoroll.nonzero()] = 1
        onoff = np.zeros(pianoroll.shape) 
        for i in range(pianoroll.shape[0]):
            if i == 0:
                onoff[i][pianoroll[i].nonzero()] = 1
            else:
                onoff[i][np.setdiff1d(pianoroll[i-1].nonzero(), pianoroll[i].nonzero())] = -1
                onoff[i][np.setdiff1d(pianoroll[i].nonzero(), pianoroll[i-1].nonzero())] = 1 
        pianoroll = np.transpose(pianoroll, (1, 0))
        onoff = np.transpose(onoff, (1, 0))
        
        # process audio
        audio, sr = librosa.load(audio_filename, sr=preprocess_hp.sr)
        spec = process_spectrum_from_chunk(audio)
        #spec = librosa.stft(audio, n_fft=process_hp.n_fft, hop_length=process_hp.stride)
        #magnitude = np.log1p(np.abs(spec)**2)

        # convert to Tensors
        pianoroll = torch.cuda.FloatTensor(pianoroll).unsqueeze(0)
        onoff = torch.cuda.FloatTensor(onoff).unsqueeze(0)
        spec = torch.cuda.FloatTensor(spec).unsqueeze(0)
        #X = torch.Tensor(score)
        #y = torch.Tensor(spec)

        # reshape into a batch with proper dims
        # TODO: Make this better...
        #pianoroll = pianoroll[:128, :860]
        #onoff = onoff[:128, :860]
        #spec = spec[:1025, :860]
        return pianoroll, onoff, spec


    def inference(self):
        score, onoff, spec = self.process_custom_midi_and_audio(self.midi_source, self.audio_source)

        model = PerformanceNet().cuda()
        model.load_state_dict(self.checkpoint['state_dict'])
                   
        print ('Inferencing spectrogram......')

        with torch.no_grad():
            model.eval()    
            test_results = model(score, spec, onoff)
            test_results = test_results.cpu().numpy()
 
        output_dir = self.create_output_dir()

        for i in range(len(test_results)):
            audio = self.griffinlim(test_results[i], audio_id = i+1)
            sf.write(os.path.join(output_dir,'output-{}.wav'.format(i+1)), audio, self.sample_rate)
    
    def create_output_dir(self):
        success = False
        dir_id = 1
        while not success:
            try:
                audio_out_dir = os.path.join(self.exp_dir,'audio_output_{}'.format(dir_id))
                os.makedirs(audio_out_dir)
                success = True
            except FileExistsError:
                dir_id += 1
        return audio_out_dir

    def griffinlim(self, spectrogram, audio_id, n_iter = 300, window = 'hann', n_fft = 2048, hop_length = 256, verbose = False):
        
        print ('Synthesizing audio {}'.format(audio_id))

        if hop_length == -1:
            hop_length = n_fft // 4
            spectrogram[0:5] = 0

        spectrogram[150:] = 0
        angles = np.exp(2j * np.pi * np.random.rand(*spectrogram.shape))

        t = tqdm(range(n_iter), ncols=100, mininterval=2.0, disable=not verbose)
        for i in t:
            full = np.abs(spectrogram).astype(np.complex) * angles
            inverse = librosa.istft(full, hop_length = hop_length, window = window)
            rebuilt = librosa.stft(inverse, n_fft = n_fft, hop_length = hop_length, window = window)
            angles = np.exp(1j * np.angle(rebuilt))

            if verbose:
                diff = np.abs(spectrogram) - np.abs(rebuilt)
                t.set_postfix(loss=np.linalg.norm(diff, 'fro'))

        full = np.abs(spectrogram).astype(np.complex) * angles
        inverse = librosa.istft(full, hop_length = hop_length, window = window)

        return inverse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-exp-name", type=str, required=True)
    parser.add_argument("-midi-source", type=str, required=True)
    parser.add_argument("-audio-source", type=str, required=True)
    args = parser.parse_args()

    exp_dir = os.path.join(os.path.abspath('./experiments'), args.exp_name) # which experiment to test
    with open(os.path.join(exp_dir,'hyperparams.json'), 'r') as hpfile:
        hp = json.load(hpfile)
    checkpoints = 'checkpoint-{}.tar'.format(hp['best_epoch'])
    AudioSynth = AudioSynthesizer(checkpoints, exp_dir, args.midi_source, args.audio_source) 
    AudioSynth.inference()


if __name__ == "__main__":
    main()
            

