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
    def __init__(self, in_file, seed=42, n_read=None):
        super(DatasetPreprocessRealTime, self).__init__()

        self.dataset = h5py.File(in_file, 'r')
        self.styles = [name.split('mfcc_')[1] for name in self.dataset.keys() if 'mfcc_' in name] # get styles from the data

        # load all the raw audio files
        self.audios = {key: self.dataset[key] for key in self.dataset.keys() if 'audio_' in key}

        # init specs
        self.torch_spectrogram = torchaudio.transforms.Spectrogram(n_fft=pp_hp.n_fft, hop_length=pp_hp.ws)
        melkwargs = {'hop_length': pp_hp.ws, 'n_fft': pp_hp.n_fft, }
        self.torch_mfcc = torchaudio.transforms.MFCC(sample_rate=pp_hp.sr, n_mfcc=12, melkwargs=melkwargs)
        #self.torch_melspec = torchaudio.transforms.MelSpectrogram(sample_rate=hp.sr, n_fft=hp.n_fft, hop_length=hp.ws)

        if n_read is None:
            n_read = 10000000   # large number to read everything

        self.pianoroll = self.dataset['pianoroll'][:n_read]
        self.onoff = self.dataset['onoff'][:n_read]
        self.mfccs = {}
        self.target_coords = {}
        for style in self.styles:
            print(f"loading style: {style}")
            self.mfccs[style] = self.dataset['mfcc_' + style][:n_read] 
            self.target_coords[style] = self.dataset['target_coords_' + style][:n_read] 

        self.n_data = self.pianoroll.shape[0]
        random.seed(seed)


    def __getitem__(self, index):
        '''
        The input data are the pianoroll, onoff, a random mfcc from the same style
        The output data is the spectrogram calculated on-the-fly (to save space) for the corresponding pianoroll/onoff
        '''
        # piano
        pianoroll = self.pianoroll[index]
        onoff = self.onoff[index]
        pianoroll = np.concatenate((pianoroll, onoff), axis=-1)
        pianoroll = np.transpose(pianoroll, (1, 0))

        # pick random style
        style = random.choice(self.styles)

        # random mfcc for selected style as input conditioning
        rand_index = random.randint(0, self.n_data - 1)
        #mfcc_rand = self.mfccs[style][rand_index]
        song_id_rand, chunk_begin_index_rand, chunk_end_index_rand = self.target_coords[style][rand_index].astype('int')
        audio_chunk_rand = self.audios[f'audio_{song_id_rand}_{style}'][chunk_begin_index_rand: chunk_end_index_rand]
        X_cond = self.torch_mfcc(torch.Tensor(audio_chunk_rand))

        # make target spectrogram
        song_id, chunk_begin_index, chunk_end_index = self.target_coords[style][index].astype('int')
        audio_chunk = self.audios[f'audio_{song_id}_{style}'][chunk_begin_index: chunk_end_index]
        y = self.torch_spectrogram(torch.Tensor(audio_chunk))

        if CUDA_FLAG == 1:
            X = torch.cuda.FloatTensor(pianoroll)
            #X_cond = torch.cuda.FloatTensor(mfcc_rand)
            X_cond = X_cond.to('cuda')
            y = y.to('cuda')
        else:
            X = torch.Tensor(pianoroll)
            #X_cond = torch.Tensor(mfcc_rand)
        return X, X_cond, y

    def __len__(self):
        return self.n_data


# class Dataseth5py(torch.utils.data.Dataset):
#     # https://discuss.pytorch.org/t/how-to-speed-up-the-data-loader/13740/3
#     def __init__(self, in_file, seed=42, n_read=None):
#         super(Dataseth5py, self).__init__()

#         self.dataset = h5py.File(in_file, 'r')
#         self.styles = [name for name in self.dataset.keys() if 'spec_' in name] # get styles from the data

#         # TODO: Need to determine whether normalization needs to happen or not
#         if n_read is not None:
#             self.pianoroll = self.dataset['pianoroll'][:n_read]
#             self.onoff = self.dataset['onoff'][:n_read]
#             self.specs = {}
#             for style in self.styles:
#                 print(f"loading style: {style}")
#                 self.specs[style] = self.dataset[style][:n_read] 
#         else:
#             self.pianoroll = self.dataset['pianoroll'][:]
#             self.onoff = self.dataset['onoff'][:]
#             self.specs = {}
#             for style in self.styles:
#                 print(f"loading style: {style}")
#                 self.specs[style] = self.dataset[style][:]

#         self.n_data = self.pianoroll.shape[0]
#         random.seed(seed)

#     def __getitem__(self, index):
#         '''
#         The input data are the pianoroll, onoff, a *random* spec from the same style
#         The output data is the *matching* spec for the corresponding pianoroll/onoff
#         '''
#         # piano
#         pianoroll = self.pianoroll[index]
#         onoff = self.onoff[index]
#         pianoroll = np.concatenate((pianoroll, onoff), axis=-1)
#         pianoroll = np.transpose(pianoroll, (1, 0))

#         # specs
#         style = random.choice(self.styles)
#         spec = self.specs[style][index]
#         rand_index = random.randint(0, self.n_data - 1)
#         spec_rand = self.specs[style][rand_index]

#         if CUDA_FLAG == 1:
#             X = torch.cuda.FloatTensor(pianoroll)
#             X_cond = torch.cuda.FloatTensor(spec_rand)
#             y = torch.cuda.FloatTensor(spec)
#         else:
#             X = torch.Tensor(pianoroll)
#             X_cond = torch.Tensor(spec_rand)
#             y = torch.Tensor(spec)
#         return X, X_cond, y

#     def __len__(self):
#         return self.n_data


def Process_Data(data_dir, n_train_read=None, n_test_read=None, batch_size=16):
    print("loading training data")
    train_dataset = DatasetPreprocessRealTime(data_dir + '_train.hdf5', n_read=n_train_read)
    print("loading test data")
    test_dataset = DatasetPreprocessRealTime(data_dir + '_test.hdf5', n_read=n_test_read)

    kwargs = {}
    train_loader = utils.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
    test_loader = utils.DataLoader(test_dataset, batch_size=batch_size, **kwargs)
    return train_loader, test_loader


class EngelLoss:
    def __init__(self, n_mels=128, alpha=1):
    # loss from https://arxiv.org/abs/2001.04643
    self.loss_function = nn.L1Loss()
    self.mel_scale = torchaudio.transforms.MelScale(n_mels=n_mels, sample_rate=pp_hp.sr)

    def loss(self, pred, target):
        loss_1 = self.loss_function(pred, target)
        loss_2 = self.loss_function(
            self.mel_scale(pred), 
            self.mel_scale(target)
        )
        total_loss = loss_1 + alpha * loss_2
        return total_loss


def train(model, epoch, train_loader, optimizer, iter_train_loss):
    model.train()
    train_loss = 0
    engel_loss = EngelLoss()
    for batch_idx, (data, data_cond, target) in enumerate(train_loader):        
        optimizer.zero_grad()
        split = torch.split(data, 128, dim=1)
        if CUDA_FLAG == 1:
            y_pred = model(split[0].cuda(), data_cond.cuda(), split[1].cuda())
            target = target.cuda()
        else:
            y_pred = model(split[0], data_cond, split[1]) 
        
        loss = engel_loss(y_pred, target)
        loss.backward()
        iter_train_loss.append(loss.item())
        train_loss += loss
        optimizer.step()    
         
        if batch_idx % 4 == 0:
            print ('Train Epoch: {} [{}/{} ({:.0f}%)]\t Loss: {:.6f}'.format(epoch, batch_idx * len(data), len(train_loader.dataset), 100. * batch_idx/len(train_loader), loss.item()/len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss/ len(train_loader.dataset)))
    return train_loss / len(train_loader.dataset)


def test(model, epoch, test_loader, scheduler, iter_test_loss):
    with torch.no_grad():
        model.eval()
        test_loss = 0
        engel_loss = EngelLoss()
        for idx, (data, data_cond, target) in enumerate(test_loader):
            split = torch.split(data, 128, dim=1)
            if CUDA_FLAG == 1:
                y_pred = model(split[0].cuda(), data_cond.cuda(), split[1].cuda())
                target = target.cuda()
            else:
                y_pred = model(split[0], data_cond, split[1])
            
            loss = engel_loss(y_pred, target)
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
    train_loader, test_loader = Process_Data(args.data_dir, n_train_read=args.n_train_read, n_test_read=args.n_test_read, batch_size=args.batch_size)
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
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    
    main(args)
