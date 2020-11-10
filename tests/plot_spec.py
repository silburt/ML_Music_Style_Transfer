import matplotlib.pyplot as plt
import librosa
import librosa.display
import numpy as np
import sys
sys.path.append('../preprocessing/')
from preprocess import hyperparams

hp = hyperparams()

AUDIO_FILENAME = "inputs/2308_prelude18_gentleman.wav"

def process_spectrum_from_chunk(audio_chunk):
    spec = librosa.stft(audio_chunk, n_fft=hp.n_fft, hop_length=hp.ws)
    magnitude_pnet = np.log1p(np.abs(spec)**2)
    
    magnitude_grifflim = np.abs(spec)

    # https://medium.com/analytics-vidhya/understanding-the-mel-spectrogram-fca2afa2ce53
    mel_spec = librosa.feature.melspectrogram(y=audio_chunk, sr=hp.sr, n_fft=hp.n_fft, hop_length=hp.ws)
    #magnitude = librosa.feature.mfcc(y=audio_chunk, sr=hp.sr, S=None, n_mfcc=20)
    return magnitude_pnet, magnitude_grifflim, mel_spec


if __name__ == '__main__':
    audio, sr = librosa.load(AUDIO_FILENAME, sr=hp.sr)
    
    # get audio chunk
    step = 0
    n_samples_per_chunk = (hp.spc * hp.wps - 1) * hp.ws
    audio_chunk = audio[(step * hp.ws * hp.stride): (step * hp.ws * hp.stride) + n_samples_per_chunk]

    # get magnitudes
    magnitude_pnet, magnitude_grifflim, mel_spec = process_spectrum_from_chunk(audio_chunk)
    print("Shapes: pnet, grifflim, mel_spec:", magnitude_pnet.shape, magnitude_grifflim.shape, mel_spec.shape)

    # plot
    fig, ax = plt.subplots(3, 1, sharex=True)
    librosa.display.specshow(magnitude_pnet, sr=hp.sr, hop_length=hp.ws, y_axis='log', x_axis='time', ax=ax[0])
    librosa.display.specshow(magnitude_grifflim, sr=hp.sr, hop_length=hp.ws, y_axis='log', x_axis='time', ax=ax[1])
    librosa.display.specshow(mel_spec, sr=hp.sr, hop_length=hp.ws, y_axis='mel', x_axis='time', ax=ax[2])
    ax[0].set_title("pnet")
    ax[1].set_title("grifflim")
    ax[2].set_title("mel_spec")
    print(f"shapes: original (pnet)={magnitude_pnet.shape}, grifflim_compat={magnitude_grifflim.shape}, mel_spec={mel_spec.shape}")
    #plt.show()
    plt.savefig("outputs/plot_spec.png")