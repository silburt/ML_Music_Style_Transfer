import pytest
import librosa
import soundfile as sf
import os
import sys
import torch
import torchaudio
sys.path.append('../model/')
sys.path.append('../preprocessing/')
from preprocess import process_audio_into_chunks, hyperparams


AUDIO_FILENAME = "2308_prelude18_harpsichord.wav"
hp = hyperparams()

def test_griffinlim():
    # prepare spectrum from audio chunk
    audio, sr = librosa.load(os.path.join("inputs", AUDIO_FILENAME), sr=hp.sr)
    song_id = AUDIO_FILENAME.split("_")[0]
    style_id = AUDIO_FILENAME.split("_")[-1].split('.wav')[0]
    spec_list, target_list = process_audio_into_chunks(audio, style_id, song_id, 1)

    # reverse - write back to audio file
    inverse = librosa.griffinlim(target_list[0], n_iter=300, window='hann', win_length=hp.n_fft, hop_length=hp.ws)
    #inverse = librosa.feature.inverse.mel_to_audio(spec_list[0], n_iter=300, sr=hp.sr, n_fft=hp.n_fft, hop_length=hp.ws)

    # write
    sf.write(f'outputs/test_griffinlim_{song_id}_{style_id}_sr{hp.sr}.wav', inverse, hp.sr)


# Seems to work really well!!
def test_torchgriffinlim():
    # init torch
    t_spec = torchaudio.transforms.Spectrogram(n_fft=hp.n_fft, hop_length=hp.ws)
    griffinlim = torchaudio.transforms.GriffinLim(n_fft=hp.n_fft, n_iter=300, win_length=hp.n_fft, hop_length=hp.ws)

    # load audio
    audio, sr = librosa.load(os.path.join("inputs", AUDIO_FILENAME), sr=hp.sr)
    song_id = AUDIO_FILENAME.split("_")[0]
    style_id = AUDIO_FILENAME.split("_")[-1].split('.wav')[0]

    # get spectrogram
    step = 0
    n_samples_per_chunk = (hp.spc * hp.wps - 1) * hp.ws
    audio_chunk = audio[(step * hp.ws * hp.stride): (step * hp.ws * hp.stride) + n_samples_per_chunk]
    spec = t_spec(torch.Tensor(audio_chunk))

    # test normalization - so it seems that log1p or normalizing the spectrogram degrades griffin-lim quite a bit
    # SO you just apply these transformations when calculating the loss, not the raw model prediction
    #spec -= spec.min(1, keepdim=True)[0]
    #spec /= spec.max(1, keepdim=True)[0]
    #spec = torch.log1p(spec)
    #print("min/max values in spec", torch.min(spec), torch.max(spec))

    # convert back to audio
    inverse = griffinlim(spec).detach().cpu().numpy()
    sf.write(f'outputs/test_griffinlim_{song_id}_{style_id}_sr{hp.sr}_torch.wav', inverse, hp.sr)


if __name__ == '__main__':
    #test_griffinlim()
    test_torchgriffinlim()