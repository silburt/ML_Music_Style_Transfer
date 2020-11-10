import pretty_midi
import os
#from pretty_midi_roll_to_midi import piano_roll_to_pretty_midi
from utils.pretty_midi_roll_to_midi import piano_roll_to_pretty_midi
import soundfile as sf
import argparse


# string to bool
def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


# mostly for debugging
def write_chunked_samples(out_dir, song_id, step, hp, style=None, audio_chunk=None, pianoroll_chunk=None):
    '''
    This writes the audio and/or pianoroll chunk to wav/midi so that you can listen/confirm that the 
    chunks are aligning properly.
    '''
    if audio_chunk is not None:
        outpath = os.path.join(out_dir, f"{song_id}_{style}_c{step}")
        sf.write(outpath + ".wav", audio_chunk, hp.sr)

    if pianoroll_chunk is not None:
        outpath = os.path.join(out_dir, f"{song_id}_c{step}")
        velocity_midi_max = 127
        reformatted_pianoroll_chunk = pianoroll_chunk.T.astype('int') * velocity_midi_max
        midi_slice = piano_roll_to_pretty_midi(reformatted_pianoroll_chunk, fs=hp.wps)
        midi_slice.write(outpath + ".mid")


class h5pyManager():
    '''
    Manages writing data to h5py. The indexes should line up properly such that pianoroll[i]/onoff[i]/spec_{style}[i] match.
    '''
    def __init__(self, data):
        self.data = data

    def write_pianoroll(self, pianoroll_list, onoff_list):
        '''
        Create and incrementally add to an h5py file, so you don't have to fit everything into memory
        https://stackoverflow.com/questions/47072859/how-to-append-data-to-one-specific-dataset-in-a-hdf5-file-with-h5py
        '''
        # TODO: find out elegant chunking - different songs have different numbers of chunks
        if 'pianoroll' not in self.data:
            # create
            self.data.create_dataset("pianoroll", data=pianoroll_list, dtype='float32', maxshape=(None,) + pianoroll_list.shape[1:], chunks=True) 
            self.data.create_dataset("onoff", data=onoff_list, dtype='float32', maxshape=(None,) + onoff_list.shape[1:], chunks=True) 
        else:
            # append
            self.data["pianoroll"].resize(self.data["pianoroll"].shape[0] + pianoroll_list.shape[0], axis=0)
            self.data["pianoroll"][-pianoroll_list.shape[0]:] = pianoroll_list

            self.data["onoff"].resize(self.data["onoff"].shape[0] + onoff_list.shape[0], axis=0)
            self.data["onoff"][-onoff_list.shape[0]:] = onoff_list

    def write_spectrum(self, spec_list, target_list, style):
        '''
        Create and incrementally add to an h5py file, so you don't have to fit everything into memory
        '''
        key_name = f'spec_{style}'
        target_name = f'target_{style}'
        if key_name not in self.data:
            # create
            self.data.create_dataset(key_name, data=spec_list, dtype='float32', maxshape=(None,) + spec_list.shape[1:], chunks=True)
            self.data.create_dataset(target_name, data=target_list, dtype='float32', maxshape=(None,) + target_list.shape[1:], chunks=True) 
        else:
            # append
            self.data[key_name].resize(self.data[key_name].shape[0] + spec_list.shape[0], axis=0)
            self.data[key_name][-spec_list.shape[0]:] = spec_list

            self.data[target_name].resize(self.data[target_name].shape[0] + target_list.shape[0], axis=0)
            self.data[target_name][-target_list.shape[0]:] = target_list

