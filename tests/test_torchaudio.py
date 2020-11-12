import pytest
import librosa
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

def test_torchaudio_transforms():
    # init
    audio, sr = librosa.load(os.path.join("inputs", AUDIO_FILENAME), sr=hp.sr)
    t_spec = torchaudio.transforms.Spectrogram(n_fft=hp.n_fft, hop_length=hp.ws)
    melkwargs = {'hop_length': hp.ws, 'n_fft': hp.n_fft}
    t_mfcc = torchaudio.transforms.MFCC(sample_rate=hp.sr, n_mfcc=12, melkwargs=melkwargs)

    # test
    for step in range(3):
        n_samples_per_chunk = (hp.spc * hp.wps - 1) * hp.ws
        audio_chunk = audio[(step * hp.ws * hp.stride): (step * hp.ws * hp.stride) + n_samples_per_chunk]

        # spec
        spec = np.abs(librosa.stft(audio_chunk, n_fft=hp.n_fft, hop_length=hp.ws)) ** 2
        torch_spec = t_spec(torch.Tensor(audio_chunk)).detach().cpu().numpy()
        assert spec.shape == torch_spec.shape
        assert np.allclose(spec, torch_spec, atol=1e-4)

        # mfcc
        mfcc = librosa.feature.mfcc(y=audio_chunk, sr=hp.sr, S=None, n_mfcc=12)
        torch_mfcc = t_mfcc(torch.Tensor(audio_chunk)).detach().cpu().numpy()
        print(mfcc.shape, torch_mfcc.shape)
        print(torch_mfcc)
        print(mfcc)
        #assert np.allclose(mfcc, torch_mfcc)
    
    print("test passes!")


if __name__ == '__main__':
    test_torchaudio_transforms()