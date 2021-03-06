import pytest
import librosa
import soundfile as sf
import sys
sys.path.append('../model/')
sys.path.append('../preprocessing/')
from preprocess import process_audio_into_chunks, hyperparams
from inference import AudioSynthesizer, _griffinlim

hp = hyperparams()

AUDIO_FILENAME = "inputs/2308_prelude18_harpsichord.wav"

def test_griffinlim():
    # prepare spectrum from audio chunk
    audio, sr = librosa.load(AUDIO_FILENAME, sr=hp.sr)
    song_id = AUDIO_FILENAME.split("_")[0]
    style_id = AUDIO_FILENAME.split("_")[-1].split('.wav')[0]
    spec_list = process_audio_into_chunks(audio, style_id, song_id, 1)

    # reverse - write to audio file
    #inverse = _griffinlim(spec_list[0], song_id)
    inverse = librosa.griffinlim(spec_list[0], n_iter=300, window='hann', win_length=hp.n_fft, hop_length=hp.ws)
    #inverse = librosa.feature.inverse.mel_to_audio(spec_list[0], n_iter=300, sr=hp.sr, n_fft=hp.n_fft, hop_length=hp.ws)

    # write
    sf.write(f'outputs/test_griffinlim_{song_id}_{style_id}.wav', inverse, hp.sr)
    
if __name__ == '__main__':
    test_griffinlim()