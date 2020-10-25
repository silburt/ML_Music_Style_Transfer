# The purpose of this script is to determine which files from the musicnet dataset are isolated piano.
# This could really be generalized to solo instruments

import glob
import pandas as pd
import os

PIANO_INSTRUMENT_LABEL = 1

def main(path_to_musicnet, data_type, output_file_basename):
    path = os.path.join(path_to_musicnet, f"{data_type}_labels")
    label_files = glob.glob(f"{path}/*.csv")
    
    wav_file_list = []
    for file in label_files:
        label = pd.read_csv(file)
        instruments = list(set(label['instrument'].values))
        if len(instruments) == 1 and instruments[0] == PIANO_INSTRUMENT_LABEL:
            wav_file_list.append(file)
    
    with open(output_file_basename + f"_{data_type}.txt", 'w') as f:
        for item in wav_file_list:
            csv_file = os.path.basename(item)
            f.write(f"{csv_file}\n")
            

if __name__ == '__main__':
    path_to_musicnet = '../data/musicnet/'
    data_type = 'test' # train or test
    output_file_basename = 'piano_pieces'
    
    main(path_to_musicnet, data_type, output_file_basename)
    
