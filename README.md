# ML_Music_Style_Transfer
Style transfer in the piano space only.

## Design
Use a Wavenet autoencoder/decoder structure with a condition on pitch (i.e. midi). From [Neural Audio Synthesis of Musical Notes with WaveNet Autoencoders](https://arxiv.org/abs/1704.01279) this conditioning happens by concatenating the latent embedding with a one-hot vector of the pitch. For music where MIDI is not available, you can get it using [onset and frames transcription model](https://github.com/magenta/magenta/tree/master/magenta/models/onsets_frames_transcription). 

Having the midi as an input to the decoder is key because then the decoder becomes "genreal purpose". I.e. in [A Universal Music Translation Network](https://github.com/facebookresearch/music-translation/tree/master/src) the style transfer happens on a decoder-by-decoder basis - to get a generic piano sound you use the piano decoder. However instead I want different kinds of piano sounds, conditioned on the inputs: a) the piano sound I provided and b) the midi input I provide. Hopefully, midi also allows polyphonic.

### Data
#### Converting midi -> vectors
As stated in the [MAESTRO](https://arxiv.org/abs/1810.12247) paper: "The input to the context stack is an onset “piano roll” representation, a size-88 vector signaling the onset of any keys on the keyboard, with 4ms bins (250Hz). Each element of the vector is a float that represents the strike velocity of a piano key in the 4ms frame, scaled to the range [0, 1]. When there
is no onset for a key at a given time, the value is 0." So I think basically you go through every midi file with a bin size and vectorize everything. Then you can input them to Wavenet. 

### Loss
However I think the loss you want to eventually use is the Spectral loss like what is done in [DDSP](https://arxiv.org/pdf/2001.04643.pdf). The point they show (and also "Phase Invariance" example [here](https://storage.googleapis.com/ddsp/index.html)) is that maximum likelihood/cross entropy is not a good loss as similarly sounding audio can have very different waveforms. 

## Specific Architectures
- [A Universal Music Translation Network](https://github.com/facebookresearch/music-translation/tree/master/src) in pytorch.
- [MAESTRO paper](https://arxiv.org/abs/1810.12247) - they say: *WaveNet (van den Oord et al., 2016) is able to synthesize realistic instrument sounds directly in the waveform domain, but it is not as adept at capturing musical structure at timescales of seconds or longer. However, if we provide a MIDI sequence to a WaveNet model as conditioning information, we eliminate the need for capturing large scale structure, and the model can focus on local structure instead, i.e., instrument timbre and local interactions between notes.*

### Additional Code Resources
- wavenet keras - https://github.com/peustr/wavenet/tree/master/wavenet
- https://github.com/basveeling/wavenet
- Temporal convolutional network - https://github.com/philipperemy/keras-tcn
- https://www.kaggle.com/christofhenkel/temporal-cnn
- https://www.kaggle.com/siavrez/wavenet-keras


### Datasets
- [Saarland Music Data (SMD)](http://resources.mpi-inf.mpg.de/SMD/SMD_MIDI-Audio-Piano-Music.html)


### Blog resources
- [example of using onset frames transcription](https://medium.com/nomtek/machine-learning-in-music-transcription-354b9360cd5f)
- [midi binning](https://raphaellederman.github.io/articles/musicgeneration/#training-the-language-model)
- [analyticsvidhya music generation](https://www.analyticsvidhya.com/blog/2020/01/how-to-perform-automatic-music-generation/)
- [Sander Dielemann blog post](https://benanne.github.io/2020/03/24/audio-generation.html)
