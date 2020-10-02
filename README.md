# ML_Music_Style_Transfer
Style transfer in the piano space only.

## Design
Use a Wavenet autoencoder/decoder structure with a condition on pitch (i.e. midi). From [Neural Audio Synthesis of Musical Notes with WaveNet Autoencoders](https://arxiv.org/abs/1704.01279) this conditioning happens by concatenating the latent embedding with a one-hot vector of the pitch. For music where MIDI is not available, you can get it using [onset and frames transcription model](https://github.com/magenta/magenta/tree/master/magenta/models/onsets_frames_transcription). 

Having the midi as an input to the decoder is key because then the decoder becomes "genreal purpose". I.e. in [A Universal Music Translation Network](https://github.com/facebookresearch/music-translation/tree/master/src) the style transfer happens on a decoder-by-decoder basis - to get a generic piano sound you use the piano decoder. However instead I want different kinds of piano sounds, conditioned on the inputs: a) the piano sound I provided and b) the midi input I provide. Hopefully, midi also allows polyphonic.

### Loss
However I think the loss you want to eventually use is the Spectral loss like what is done in [DDSP](https://arxiv.org/pdf/2001.04643.pdf). The point they show (and also "Phase Invariance" example [here](https://storage.googleapis.com/ddsp/index.html)) is that maximum likelihood/cross entropy is not a good loss as similarly sounding audio can have very different waveforms. 

## Specific Architectures
- [A Universal Music Translation Network](https://github.com/facebookresearch/music-translation/tree/master/src) in pytorch.

### Additional Code Resources
- wavenet keras - https://github.com/peustr/wavenet/tree/master/wavenet
- https://github.com/basveeling/wavenet
- Temporal convolutional network - https://github.com/philipperemy/keras-tcn
- https://www.kaggle.com/christofhenkel/temporal-cnn
- https://www.kaggle.com/siavrez/wavenet-keras


### Blog resources
- [example of using onset frames transcription](https://medium.com/nomtek/machine-learning-in-music-transcription-354b9360cd5f)
