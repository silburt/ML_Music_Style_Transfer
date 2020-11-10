import matplotlib.pyplot as plt
import librosa
import librosa.display
import numpy as np

AUDIO_BASENAME = "inputs/2308_prelude18_"
STYLES = ['aliciakeys', 'cuba', 'gentleman', 'harpsichord', 'markisuitcase', 'upright']

class hyperparams(object):
    '''
    Definitions:
        - window: a pianoroll column / unit of time
        - chunk: the entire pianoroll segment that will constitude a data point (X windows make a chunk)
    '''
    def __init__(self):
        self.sr = 44100 // 2 # Sampling rate (samples per second)
        self.n_fft = 2048 # fft points (samples) 2048
        self.stride = 512 # number of windows of separation between chunks/data points
        
        # A.S. each song is chopped into windows, and I *think* hop is the window length?
        self.ws = 256   # window size (audio samples per window)
        self.wps = 44100 // self.ws # ~172 windows/second
        self.spc = 5    # seconds per chunk

hp = hyperparams()

def preprocess(audio_chunk, preprocess_feature):
    if preprocess_feature == 'stft':
        spec = librosa.stft(audio_chunk, n_fft=hp.n_fft, hop_length=hp.ws)
        magnitude = np.log1p(np.abs(spec)**2)
    elif preprocess_feature == 'mel':
        magnitude = librosa.feature.melspectrogram(y=audio_chunk, sr=hp.sr, n_fft=hp.n_fft, hop_length=hp.ws)
    elif preprocess_feature == 'mfcc':
        magnitude = librosa.feature.mfcc(y=audio_chunk, sr=hp.sr, S=None, n_mfcc=13)
    print("magnitude shape:", magnitude.shape)
    return magnitude


def get_audio_chunk(audio):
    # get audio chunk
    step = 0
    n_samples_per_chunk = (hp.spc * hp.wps - 1) * hp.ws
    audio_chunk = audio[(step * hp.ws * hp.stride): (step * hp.ws * hp.stride) + n_samples_per_chunk]
    return audio_chunk


def main(preprocess_feature):
    fig, ax = plt.subplots(len(STYLES), 1, sharex=True, figsize=(6,9))
    for i, style in enumerate(STYLES):
        audio, sr = librosa.load(AUDIO_BASENAME + style + ".wav", sr=hp.sr)
        audio_chunk = get_audio_chunk(audio)
        magnitude = preprocess(audio_chunk, preprocess_feature)
        librosa.display.specshow(magnitude, sr=hp.sr, hop_length=hp.ws, y_axis='linear', ax=ax[i])
        ax[i].set_title(style, fontsize=8)
    title = f"{preprocess_feature}, sr={hp.sr}, shape={magnitude.shape}, n_fft={hp.n_fft}, hop_length={hp.ws}"
    plt.suptitle(title)
    #plt.show()
    plt.savefig(f"outputs/plot_styles_{preprocess_feature}_sr{hp.sr}_nfft{hp.n_fft}.png")

if __name__ == '__main__':
    preprocess_feature = 'mfcc' # mfcc, mel, stft
    main(preprocess_feature)