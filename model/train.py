'''
function ClickConnect(){
console.log("Clicking");
document.querySelector("colab-connect-button").click()
}
setInterval(ClickConnect,100000)
'''

import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import torchaudio
import librosa
import numpy as np
import torch.optim as optim
from sklearn.model_selection import train_test_split
import torch.utils.data as utils
import h5py
import sys
import os
import json
from model import PerformanceNet
import argparse
import os
import time
import random
import sys
sys.path.append('../preprocessing')
from preprocess import hyperparams as pp_hyperparams

pp_hp = pp_hyperparams()

CUDA_FLAG = 0
if torch.cuda.is_available():
    cuda = torch.device("cuda")
    CUDA_FLAG = 1

class hyperparams(object):
    def __init__(self, args):
        self.train_epoch = args.epochs
        self.test_freq = args.test_freq
        self.exp_name = args.exp_name
        self.iter_train_loss = []
        self.iter_test_loss = []
        self.loss_history = []
        self.test_loss_history = []
        self.best_loss = 1e10 
        self.best_epoch = 0


class DatasetPreprocessRealTime(torch.utils.data.Dataset):
    def __init__(self, in_file, seed=42, n_read=None, n_spec_precal=500):
        super(DatasetPreprocessRealTime, self).__init__()

        self.dataset = h5py.File(in_file, 'r')
        self.styles = [name.split('target_coords_')[1] for name in self.dataset.keys() if 'target_coords_' in name] # get styles from the data

        # load all the raw audio files
        self.audios = {key: self.dataset[key] for key in self.dataset.keys() if 'audio_' in key}

        # init specs
        self.torch_spectrogram = torchaudio.transforms.Spectrogram(n_fft=pp_hp.n_fft, hop_length=pp_hp.ws)
        #melkwargs = {'hop_length': pp_hp.ws, 'n_fft': pp_hp.n_fft, }
        #self.torch_mfcc = torchaudio.transforms.MFCC(sample_rate=pp_hp.sr, n_mfcc=16, melkwargs=melkwargs)
        #self.torch_melspec = torchaudio.transforms.MelSpectrogram(sample_rate=pp_hp.sr, n_fft=pp_hp.n_fft, hop_length=pp_hp.ws)

        if n_read is None:
            n_read = 10000000   # large number to read everything

        self.pianoroll = self.dataset['pianoroll'][:n_read]
        self.onoff = self.dataset['onoff'][:n_read]
        #self.mfccs = {}
        self.target_coords = {}
        for style in self.styles:
            print(f"loading style: {style}")
            self.target_coords[style] = self.dataset['target_coords_' + style][:n_read] 

        self.n_data = self.pianoroll.shape[0]
        random.seed(seed)

        self.spec_precal = None
        if n_spec_precal is not None:
            n_spec_precal = min(self.n_data, n_spec_precal)
            print(f"Precalculating {n_spec_precal} specs to store in memory")
            spec_precal = []
            for index in range(n_spec_precal):
                _, _, audio_chunk_rand, _ = self.select_piano_and_audio_chunks(index)
                spec_precal.append(self._calc_input_conditioning(audio_chunk_rand))
            self.spec_precal = spec_precal
            self.n_spec_precal = n_spec_precal


    def select_piano_and_audio_chunks(self, index):
       # piano
        pianoroll = self.pianoroll[index]
        onoff = self.onoff[index]

        # pick random style
        style = random.choice(self.styles)

        # random audio_chunk of selected style for input conditioning
        rand_index = random.randint(0, self.n_data - 1)
        song_id_rand, chunk_begin_index_rand, chunk_end_index_rand = self.target_coords[style][rand_index].astype('int')
        audio_chunk_rand = self.audios[f'audio_{song_id_rand}_{style}'][chunk_begin_index_rand: chunk_end_index_rand]
        
        # get correct audio_chunk of selected style for output target
        song_id, chunk_begin_index, chunk_end_index = self.target_coords[style][index].astype('int')
        audio_chunk = self.audios[f'audio_{song_id}_{style}'][chunk_begin_index: chunk_end_index]

        return pianoroll, onoff, audio_chunk_rand, audio_chunk


    def _calc_input_conditioning(self, audio_chunk_rand):
        X_cond = torch.log1p(self.torch_spectrogram(torch.Tensor(audio_chunk_rand)))
        #X_cond = self.torch_mfcc(torch.Tensor(audio_chunk_rand))
        #X_cond = self.torch_melspec(torch.Tensor(audio_chunk_rand))
        #X_cond = librosa.feature.melspectrogram(y=audio_chunk_rand, sr=pp_hp.sr, 
        #                                        n_fft=pp_hp.n_fft, hop_length=pp_hp.ws)
        return X_cond


    def __getitem__(self, index):
        '''
        The input data are the pianoroll, onoff, a random mfcc from the same style
        The output data is the spectrogram calculated on-the-fly (to save space) for the corresponding pianoroll/onoff
        '''
        pianoroll, onoff, audio_chunk_rand, audio_chunk = self.select_piano_and_audio_chunks(index)

        # prepare pianoroll
        pianoroll = np.concatenate((pianoroll, onoff), axis=-1)
        pianoroll = np.transpose(pianoroll, (1, 0))

        # prepare input conditioning
        X_cond = self.spec_precal[index] if (self.n_spec_precal is not None and index < self.n_spec_precal) else self._calc_input_conditioning(audio_chunk_rand)

        # prepare target
        y = self.torch_spectrogram(torch.Tensor(audio_chunk))  # no log1p, done later in loss function
        #y = self.torch_spectrogram(torch.Tensor(audio_chunk))
        #y = np.square(np.abs(librosa.stft(audio_chunk, n_fft=pp_hp.n_fft, hop_length=pp_hp.ws)))

        if CUDA_FLAG == 1:
            X = torch.cuda.FloatTensor(pianoroll)
            #X_cond = torch.cuda.FloatTensor(X_cond)
            #y = torch.cuda.FloatTensor(y)
            X_cond = X_cond.to('cuda')
            y = y.to('cuda')
        else:
            X = torch.Tensor(pianoroll)
            X_cond = torch.Tensor(X_cond)
            y = torch.Tensor(y)
        return X, X_cond, y

    def __len__(self):
        return self.n_data


def Process_Data(data_dir, n_train_read=None, n_test_read=None, batch_size=16, n_train_spec_precal=1000):
    print("loading training data")
    train_dataset = DatasetPreprocessRealTime(data_dir + '_train.hdf5', n_read=n_train_read, n_spec_precal=n_train_spec_precal)
    print("loading test data")
    test_dataset = DatasetPreprocessRealTime(data_dir + '_test.hdf5', n_read=n_test_read, n_spec_precal=None)

    kwargs = {}
    train_loader = utils.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
    test_loader = utils.DataLoader(test_dataset, batch_size=batch_size, **kwargs)
    return train_loader, test_loader


class L2L1Loss:
    def __init__(self, alpha=1):
        self.l1 = nn.L1Loss()
        self.l2 = nn.MSELoss()
        self.alpha = alpha

    def __call__(self, pred, target):
        # From Engel (2017), Nsynth paper - We found that training on the log magnitude of the power spectra, 
        # peak normalized to be between 0 and 1, correlated better with perceptual distortion.
        pred = torch.log1p(torch.clamp(pred, min=0))
        target = torch.log1p(target)
        total_loss = self.l2(pred, target) + self.alpha * self.l1(pred, target)
        return total_loss


def train(model, epoch, train_loader, optimizer, iter_train_loss):
    model.train()
    train_loss = 0
    loss_function = L2L1Loss()
    #loss_function = nn.L1Loss()
    for batch_idx, (data, data_cond, target) in enumerate(train_loader):        
        optimizer.zero_grad()
        split = torch.split(data, 128, dim=1)
        if CUDA_FLAG == 1:
            y_pred = model(split[0].cuda(), data_cond.cuda(), split[1].cuda())
            target = target.cuda()
        else:
            y_pred = model(split[0], data_cond, split[1]) 
        
        loss = loss_function(y_pred, target)
        loss.backward()
        iter_train_loss.append(loss.item())
        train_loss += loss
        optimizer.step()    
         
        if batch_idx % 16 == 0:
            print ('Train Epoch: {} [{}/{} ({:.0f}%)]\t Loss: {:.6f}'.format(epoch, batch_idx * len(data), len(train_loader.dataset), 100. * batch_idx/len(train_loader), loss.item()/len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss/ len(train_loader.dataset)))
    return train_loss / len(train_loader.dataset)


def test(model, epoch, test_loader, scheduler, iter_test_loss):
    with torch.no_grad():
        model.eval()
        test_loss = 0
        #loss_function = nn.L1Loss()
        loss_function = L2L1Loss()
        for idx, (data, data_cond, target) in enumerate(test_loader):
            split = torch.split(data, 128, dim=1)
            if CUDA_FLAG == 1:
                y_pred = model(split[0].cuda(), data_cond.cuda(), split[1].cuda())
                target = target.cuda()
                y_pred = y_pred.cuda()
            else:
                y_pred = model(split[0], data_cond, split[1])
            
            loss = loss_function(y_pred, target)
            iter_test_loss.append(loss.item())
            test_loss += loss    
        test_loss/= len(test_loader.dataset)
        scheduler.step(test_loss)
        print ('====> Test set loss: {:.4f}'.format(test_loss))
        return test_loss


def main(args):
    hp = hyperparams(args)

    try:
        exp_root = os.path.join(os.path.abspath('./'),'experiments')
        os.makedirs(exp_root)
    except FileExistsError:
        pass
    
    exp_dir = os.path.join(exp_root, hp.exp_name)
    os.makedirs(exp_dir)

    model = PerformanceNet()
    if CUDA_FLAG == 1:
        model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    model.zero_grad()
    optimizer.zero_grad()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    train_loader, test_loader = Process_Data(args.data_dir, n_train_read=args.n_train_read, n_test_read=args.n_test_read, 
                                             batch_size=args.batch_size, n_train_spec_precal=n_train_spec_precal)
    print ('start training')
    for epoch in range(hp.train_epoch):
        loss = train(model, epoch, train_loader, optimizer, hp.iter_train_loss)
        hp.loss_history.append(loss.item())
        
        # test
        if epoch % hp.test_freq == 0:
            test_loss = test(model, epoch, test_loader, scheduler, hp.iter_test_loss)
            hp.test_loss_history.append(test_loss.item())
            if test_loss < hp.best_loss:
                print("saving model")         
                torch.save({'epoch': epoch + 1, 'state_dict': model.state_dict(), 'optimizer' : optimizer.state_dict()}, os.path.join(exp_dir, 'checkpoint-{}.tar'.format(str(epoch + 1 ))))
                hp.best_loss = test_loss.item()    
                hp.best_epoch = epoch + 1    
                with open(os.path.join(exp_dir, 'hyperparams.json'), 'w') as outfile:   
                    json.dump(hp.__dict__, outfile)
       

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-data-dir", type=str, default='/Users/arisilburt/Machine_Learning/music/PerformanceNet_ari/data/', help="directory where data is")
    parser.add_argument("-epochs", type=int, default=1)
    parser.add_argument("-test-freq", type=int, default=1, help='how many epochs between running against test data')
    parser.add_argument("-exp-name", type=str, default='piano_test')
    parser.add_argument("--n-train-read", type=int, default=None, help='How many data points to read (length of an epoch)')
    parser.add_argument("--n-test-read", type=int, default=None, help='How many data points to read (length of an epoch)')
    parser.add_argument("--n-train-spec-precal", type=int, default=1000, help='How many spectrograms to precalculate')
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    
    main(args)
