import h5py
import random
import sys
sys.path.append('../preprocessing/')
sys.path.append('../model/')
from utils.io_manager import write_chunked_samples
from train import DatasetPreprocessRealTime
from preprocess import hyperparams as pp_hyperparams

pp_hp = pp_hyperparams()

H5PY_FILE = "../preprocessing/data_products/style_transfer_train.hdf5"
N_TEST = 10
OUT_DIR = "debugdir"
DUMMY_SONG_ID = 0
DUMMY_STYLE = 'dummystyle'
random.seed(42)


def test_alignments():
    dataset = DatasetPreprocessRealTime(H5PY_FILE)
    n_data = dataset.n_data

    for i in range(N_TEST):
        rand_index = random.randint(0, n_data - 1)
        pianoroll, onoff, audio_chunk_rand, audio_chunk = dataset.select_piano_and_audio_chunks(rand_index)
        write_chunked_samples(OUT_DIR, DUMMY_SONG_ID, rand_index, pp_hp, style=DUMMY_STYLE, 
                              audio_chunk=audio_chunk, pianoroll_chunk=pianoroll)
        write_chunked_samples(OUT_DIR, DUMMY_SONG_ID, rand_index, pp_hp, style="randaudiochunk", 
                              audio_chunk=audio_chunk_rand)
        print("wrote chunk")

if __name__ == '__main__':
    test_alignments()