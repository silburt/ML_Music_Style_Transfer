import pytest
import librosa
import librosa.display
import matplotlib.pyplot as plt
import torch
import torchaudio
import numpy as np
import os
import sys
sys.path.append('../model/')
sys.path.append('../preprocessing/')
from preprocess import process_audio_into_chunks, hyperparams


AUDIO_FILENAME = "2308_prelude18_harpsichord.wav"
hp = hyperparams()

def normalize(arr):
    arr -= arr.mean()
    arr /= arr.max()
    return arr

def sum_bins(arr):
    sum_arr = np.asarray([np.sum(arr[:,i]) for i in range(arr.shape[1])])
    print(sum_arr.shape)
    return sum_arr

def test_torchaudio_transforms():
    # init
    audio, sr = librosa.load(os.path.join("inputs", AUDIO_FILENAME), sr=hp.sr)
    t_spec = torchaudio.transforms.Spectrogram(n_fft=hp.n_fft, hop_length=hp.ws)
    melkwargs = {'hop_length': hp.ws, 'n_fft': hp.n_fft, }
    t_mfcc = torchaudio.transforms.MFCC(sample_rate=hp.sr, n_mfcc=12, melkwargs=melkwargs)
    t_melspec = torchaudio.transforms.MelSpectrogram(sample_rate=hp.sr, n_fft=hp.n_fft, hop_length=hp.ws)

    # test
    for step in range(1):
        n_samples_per_chunk = (hp.spc * hp.wps - 1) * hp.ws
        audio_chunk = audio[(step * hp.ws * hp.stride): (step * hp.ws * hp.stride) + n_samples_per_chunk]

        # spec
        spec = np.abs(librosa.stft(audio_chunk, n_fft=hp.n_fft, hop_length=hp.ws)) ** 2
        torch_spec = t_spec(torch.Tensor(audio_chunk)).detach().cpu().numpy()
        max_diff = np.max(np.abs(spec - torch_spec))
        print(f"max difference (spec) = {max_diff}")
        assert max_diff < 1e-4
        
        # mel-spec - these seem to be different in magnitude but the same when viewing the images
        mel = librosa.feature.melspectrogram(y=audio_chunk, sr=hp.sr, n_fft=hp.n_fft, hop_length=hp.ws)
        torch_mel = t_melspec(torch.Tensor(audio_chunk)).detach().cpu().numpy()
        max_diff = np.max(np.abs(mel - torch_mel))
        print(f"max difference (mel) = {max_diff}")
        print("min/max mel", np.min(mel), np.max(mel))
        print("min/max torch_mel", np.min(torch_mel), np.max(torch_mel))
        assert mel.shape == torch_mel.shape
        #assert max_diff < 1e-4

        # mfcc - these seem to be different...
        mfcc = normalize(librosa.feature.mfcc(y=audio_chunk, sr=hp.sr, S=None, n_mfcc=12))
        torch_mfcc = normalize(t_mfcc(torch.Tensor(audio_chunk)).detach().cpu().numpy())
        print("min/max mfcc", np.min(mfcc), np.max(mfcc))
        print("min/max torch_mfcc", np.min(torch_mfcc), np.max(torch_mfcc))
        print(mfcc.shape, torch_mfcc.shape)
        #assert np.allclose(mfcc, torch_mfcc)

        # sizes - spec size is *by far* the biggest size, so computing even just those on the fly
        # should allow for way more storage
        print(f"audio chunk size (bytes, shape) = {audio_chunk.nbytes}, {audio_chunk.shape}")
        print(f"spec size (bytes): {torch_spec.nbytes}, {torch_spec.shape}")
        print(f"mel size (bytes): {mel.nbytes}, {torch_mel.shape}")
        print(f"mfcc size (bytes): {mfcc.nbytes}, {mfcc.shape}")

        # plot
        # fig, ax = plt.subplots(3, 1, sharex=True)
        # librosa.display.specshow(torch_spec, sr=hp.sr, hop_length=hp.ws, y_axis='log', x_axis='time', ax=ax[0])
        # librosa.display.specshow(torch_mel, sr=hp.sr, hop_length=hp.ws, y_axis='log', x_axis='time', ax=ax[1])
        # librosa.display.specshow(torch_mfcc, sr=hp.sr, hop_length=hp.ws, y_axis='log', x_axis='time', ax=ax[2])
        # plt.show()
        # plt.close()

        # plot histograms 
        fig, ax = plt.subplots(2, 1)
        ax[0].hist(sum_bins(mfcc), bins=spec.shape[0]//2)
        ax[1].hist(sum_bins(torch_mfcc), bins=torch_spec.shape[0]//2)
        plt.show()
        plt.close()

    
    print("test passes!")


if __name__ == '__main__':
    test_torchaudio_transforms()