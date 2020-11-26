# ML_Music_Style_Transfer
Style transfer in the piano space only.

## Design
Use a Wavenet autoencoder/decoder structure with a condition on pitch (i.e. midi). From [Neural Audio Synthesis of Musical Notes with WaveNet Autoencoders](https://arxiv.org/abs/1704.01279) this conditioning happens by concatenating the latent embedding with a one-hot vector of the pitch. For music where MIDI is not available, you can get it using [onset and frames transcription model](https://github.com/magenta/magenta/tree/master/magenta/models/onsets_frames_transcription). 

Having the midi as an input to the decoder is key because then the decoder becomes "genreal purpose". I.e. in [A Universal Music Translation Network](https://github.com/facebookresearch/music-translation/tree/master/src) the style transfer happens on a decoder-by-decoder basis - to get a generic piano sound you use the piano decoder. However instead I want different kinds of piano sounds, conditioned on the inputs: a) the piano sound I provided and b) the midi input I provide. Hopefully, midi also allows polyphonic.

### Data
#### Converting midi -> vectors
As stated in the [MAESTRO](https://arxiv.org/abs/1810.12247) paper: "The input to the context stack is an onset “piano roll” representation, a size-88 vector signaling the onset of any keys on the keyboard, with 4ms bins (250Hz). Each element of the vector is a float that represents the strike velocity of a piano key in the 4ms frame, scaled to the range [0, 1]. When there
is no onset for a key at a given time, the value is 0." So I think basically you go through every midi file with a bin size and vectorize everything. Then you can input them to Wavenet. 

[this repo](https://github.com/bwang514/PerformanceNet) is probably very important for you to use to make sure your midi/audio are aligning properly. How are they doing it??

**Note**: Something *very* key will be to take a single midi and convert it to many different sounding piano versions. Take the MidiNet midi files and augment them in Native Instruments. Make them available online, I am sure they will be valued. 
**Note**: The second key thing to make the conditioning properly work is to have the input audio be the same timbre but deliberately different notes (and many different midi combinations). The output spectrogram is the actual matching midi/piano. This is *easily* done via mixcraft. 

#### Upsampling 
You probably need to upsample your midi vector arrays to make sure that it has the same temporal resolution as the output audio you want. This is also what is done in [conditioning deep generative raw audio models for structured automatic music](https://arxiv.org/pdf/1806.09905.pdf).

### Loss
However I think the loss you want to eventually use is the Spectral loss like what is done in [DDSP](https://arxiv.org/pdf/2001.04643.pdf). The point they show (and also "Phase Invariance" example [here](https://storage.googleapis.com/ddsp/index.html)) is that maximum likelihood/cross entropy is not a good loss as similarly sounding audio can have very different waveforms. 

## Current Best Idea
Following [NATURAL TTS SYNTHESIS BY CONDITIONINGWAVENET ON MEL SPECTROGRAM
PREDICTIONS](https://arxiv.org/pdf/1712.05884.pdf), your encoded audio is the mel-spectrogram (potentially run through a few CNN layers), while your input symbolic representation is the midi data (probably upsampled to the same output audio sampling rate), as is mentioned in [conditioning deep generative raw audio models for structured automatic music](https://arxiv.org/pdf/1806.09905.pdf).

PerformanceNet Figure 2: pretrain the networks as end-to-end models is for sure a good idea. But then the encoder doesnt learn the right thing when you mix-n-match - it's learning a melody + timbre representation, where you just want the timbre. Instead you need to train single end-to-end with the audio E_a and midi E_s encoders, and then a single D_a decoder. How to combine the latent representations? Maybe concatenate (with a constant token always to use as a separator?) followed by dense network (to learn feature combinations) and then whatever they have as the decoder. 

I think it's also just worth trying the audio convolutions and then concatenate with the piano midi (upscale to match dimension). Then at each layer you have piano roll skip connections, upscaled to match the output dimension.

## Specific Architectures
- [A Universal Music Translation Network](https://github.com/facebookresearch/music-translation/tree/master/src) in pytorch.
- [MAESTRO paper](https://arxiv.org/abs/1810.12247) - they say: *WaveNet (van den Oord et al., 2016) is able to synthesize realistic instrument sounds directly in the waveform domain, but it is not as adept at capturing musical structure at timescales of seconds or longer. However, if we provide a MIDI sequence to a WaveNet model as conditioning information, we eliminate the need for capturing large scale structure, and the model can focus on local structure instead, i.e., instrument timbre and local interactions between notes.*
- [tacotron2](https://github.com/NVIDIA/tacotron2) with [paper](https://arxiv.org/pdf/1712.05884.pdf) - condition a wavenet decoder on mel-spectrograms. This could be what you need to do - your encoder is just audio->mel-spectrogram, and decoder is wavenet conditioned on midi + mel-spectrogram. Would be a lot simpler and probably easier to train.
- [PerformanceNet](https://github.com/bwang514/PerformanceNet) - looks like they skip wavenet altogether, and use convolutional. Easier to train apparently as wavenet is slow and data hungry. They go from score -> audio mapping. 
- [CONDITIONING DEEP GENERATIVE RAW AUDIO MODELS FOR
STRUCTURED AUTOMATIC MUSIC](https://arxiv.org/pdf/1806.09905.pdf) - this may be the ticket for what you want, and at the very least gives you a nice explanation of how to do the conditioning. 

### Additional Code Resources
- wavenet keras - https://github.com/peustr/wavenet/tree/master/wavenet
- https://github.com/basveeling/wavenet
- Temporal convolutional network - https://github.com/philipperemy/keras-tcn
- https://www.kaggle.com/christofhenkel/temporal-cnn
- https://www.kaggle.com/siavrez/wavenet-keras


### Datasets
- [Saarland Music Data (SMD)](http://resources.mpi-inf.mpg.de/SMD/SMD_MIDI-Audio-Piano-Music.html)
- [Lakh Midi Dataset](https://colinraffel.com/projects/lmd/#get)


### Blog resources
- [example of using onset frames transcription](https://medium.com/nomtek/machine-learning-in-music-transcription-354b9360cd5f)
- [midi binning](https://raphaellederman.github.io/articles/musicgeneration/#training-the-language-model)
- [analyticsvidhya music generation](https://www.analyticsvidhya.com/blog/2020/01/how-to-perform-automatic-music-generation/)
- [Sander Dielemann blog post](https://benanne.github.io/2020/03/24/audio-generation.html)
- [local vs. global conditioning github issue](https://github.com/ibab/tensorflow-wavenet/issues/112)

### Tips and tricks
- [this video](https://www.youtube.com/watch?v=Z7YM-HAz-IY&ab_channel=SethAdams) - downsample audio so that nyquist frequency is lower, and thus naturally remove some of the high frequency noise.

### Progress So Far
- Can more-or-less reproduce their results in [this branch](https://github.com/silburt/PerformanceNet/tree/orig-plus-mods) - only modification is generating spectrograms at runtime which makes things slower but allows everything to fit in memory
- Most successful attempt so far is [pre_spec-unet branch](https://github.com/silburt/ML_Music_Style_Transfer/tree/pre_spec-unet), but my hunch is that there have to be better options.

#### To try next
- Add the improvements of mel-spec branch to pre_spec-unet branch - namely loading on-the-fly so you can use all the data
- add a normalization layer?
- look into adding conditioning to the texture net piece? 
- look into different architectures...
