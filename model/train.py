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

LARGE_NUMBER = 10000000

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


class AudioTransformations:
    def __init__(self):
        # init audio transformations
        self.transformations = {
            'from_spec': {
                'spec': lambda x: x,    # pass in a spec, just return it
                'mel': torchaudio.transforms.MelScale(sample_rate=pp_hp.sr, f_min=20),
            },
            'from_audio': {
                'spec': torchaudio.transforms.Spectrogram(n_fft=pp_hp.n_fft, hop_length=pp_hp.ws),
                'mel': torchaudio.transforms.MelSpectrogram(sample_rate=pp_hp.sr, n_fft=pp_hp.n_fft, hop_length=pp_hp.ws),
                'mfcc': torchaudio.transforms.MFCC(sample_rate=pp_hp.sr, n_mfcc=16, melkwargs={'hop_length': pp_hp.ws, 'n_fft': pp_hp.n_fft, })
            }
        }
    def __call__(self, from_str, type_str):
        return self.transformations[from_str][type_str]

audio_transformations = AudioTransformations()


class DatasetPreprocessRealTime(torch.utils.data.Dataset):
    def __init__(self, in_file, seed=42, n_read=None, n_spec_precal=None, input_cond='mel'):
        super(DatasetPreprocessRealTime, self).__init__()

        self.dataset = h5py.File(in_file, 'r')
        self.styles = [name.split('target_coords_')[1] for name in self.dataset.keys() if 'target_coords_' in name] # get styles from the data

        # load all audio files and prepare transformations
        self.audios = {key: self.dataset[key] for key in self.dataset.keys() if 'audio_' in key}
        print(f"using input conditioning: {input_cond}")
        self.input_conditioning_from_spec = audio_transformations('from_spec', input_cond)
        self.input_conditioning_from_audio = audio_transformations('from_audio', input_cond)
        self.spec_transformation_from_audio = audio_transformations('from_audio', 'spec')

        # load data from h5py
        if n_read is None:
            n_read = LARGE_NUMBER   # large number to read everything

        self.pianoroll = self.dataset['pianoroll'][:n_read]
        self.onoff = self.dataset['onoff'][:n_read]
        self.target_coords = {}
        for style in self.styles:
            print(f"loading style: {style}")
            self.target_coords[style] = self.dataset['target_coords_' + style][:n_read] 
        self.n_data = self.pianoroll.shape[0]
        random.seed(seed)

        # pre-calculate the input conditioning and output spec to save time
        self.spec_precal = None
        self.n_spec_precal = n_spec_precal
        if self.n_spec_precal is not None:
            print(f"Precalculating {n_spec_precal} specs to store in memory")
            self.n_spec_precal = self.n_data if self.n_spec_precal == -1 else min(self.n_data, self.n_spec_precal)
            self.spec_precal = {}
            for style in self.styles:
                print(f"style: {style}")
                self.spec_precal[style] = []
                for index in range(self.n_spec_precal):
                    audio_chunk = self._get_audio_chunk(style, index)
                    self.spec_precal[style].append(self.spec_transformation_from_audio(torch.Tensor(audio_chunk)))
            print(f"Precalculated {n_spec_precal} specs")

        # get input cond dim
        _X_cond, _y = self.get_audio_conditioning_and_target(0, 0, self.styles[0])
        self.input_cond_dim = _X_cond.shape[0]


    def _get_audio_chunk(self, style, index):
        song_id, chunk_begin_index, chunk_end_index = self.target_coords[style][index].astype('int')
        audio_chunk = self.audios[f'audio_{song_id}_{style}'][chunk_begin_index: chunk_end_index]
        return audio_chunk


    def _calc_input_conditioning(self, audio_chunk=None, spec=None):
        if spec is not None:
            # apply transformation on spec
            X_cond = self.input_conditioning_from_spec(spec)
        elif audio_chunk is not None:
            # calc from audio signal
            X_cond = self.input_conditioning_from_audio(torch.Tensor(audio_chunk))
            #X_cond = self.torch_spectrogram(torch.Tensor(audio_chunk_rand))
            # X_cond = torch.log1p(self.torch_spectrogram(torch.Tensor(audio_chunk_rand)))
            # X_cond = self.torch_mfcc(torch.Tensor(audio_chunk_rand))
            # X_cond = self.torch_melspec(torch.Tensor(audio_chunk_rand))
            # X_cond = librosa.feature.melspectrogram(y=audio_chunk_rand, sr=pp_hp.sr, 
            #                                        n_fft=pp_hp.n_fft, hop_length=pp_hp.ws)
        else:
            raise ValueError("audio chunk and spec cant both be None")
        return X_cond


    def get_audio_conditioning_and_target(self, index, rand_index, style):
        # target
        if self.n_spec_precal is not None and index < self.n_spec_precal:
            # use pre-calculated values
            y = self.spec_precal[style][index]
        else:
            audio_chunk = self._get_audio_chunk(style, index)
            y = self.spec_transformation_from_audio(torch.Tensor(audio_chunk))

        # input condition
        if self.n_spec_precal is not None and rand_index < self.n_spec_precal:
            # NOTE: this needs to be changed if input contioning is no longer spec
            spec_rand = self.spec_precal[style][rand_index]
            X_cond = self._calc_input_conditioning(spec=spec_rand)
        else:
            audio_chunk_rand = self._get_audio_chunk(style, rand_index)
            X_cond = self._calc_input_conditioning(audio_chunk=audio_chunk_rand)
        return X_cond, y


    def get_pianoroll(self, index):
        # prepare pianoroll
        pianoroll = self.pianoroll[index]
        onoff = self.onoff[index]

        pianoroll = np.concatenate((pianoroll, onoff), axis=-1)
        pianoroll = np.transpose(pianoroll, (1, 0))
        return pianoroll


    def __getitem__(self, index):
        '''
        The input data are the pianoroll, onoff, a random mfcc from the same style
        The output data is the spectrogram calculated on-the-fly (to save space) for the corresponding pianoroll/onoff
        '''
        # get pianoroll
        pianoroll = self.get_pianoroll(index)

        # prepare audio input / target
        style = random.choice(self.styles)
        rand_index = random.randint(0, self.n_data - 1)
        X_cond, y = self.get_audio_conditioning_and_target(index, rand_index, style)

        # cudify
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


def Process_Data(data_dir, n_train_read=None, n_test_read=None, batch_size=16, n_train_spec_precal=1000, input_cond='spec'):
    print("loading training data")
    train_dataset = DatasetPreprocessRealTime(data_dir + '_train.hdf5', n_read=n_train_read, n_spec_precal=n_train_spec_precal, input_cond=input_cond)
    input_cond_dim = train_dataset.input_cond_dim

    print("loading test data")
    test_dataset = DatasetPreprocessRealTime(data_dir + '_test.hdf5', n_read=n_test_read, n_spec_precal=None, input_cond=input_cond)

    kwargs = {}
    train_loader = utils.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
    test_loader = utils.DataLoader(test_dataset, batch_size=batch_size, **kwargs)

    return train_loader, test_loader, input_cond_dim


class L2L1Loss:
    def __init__(self, alpha=1):
        self.l1 = nn.L1Loss()
        self.l2 = nn.MSELoss()
        self.melscale = audio_transformations('from_spec', 'mel')
        self.alpha = alpha

    def __call__(self, pred, target):
        # From Engel (2017), Nsynth paper - We found that training on the log magnitude of the power spectra, 
        # peak normalized to be between 0 and 1, correlated better with perceptual distortion.
        pred = torch.log1p(self.melscale(pred))
        target = torch.log1p(self.melscale(target))
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

    train_loader, test_loader, input_cond_dim = Process_Data(args.data_dir, n_train_read=args.n_train_read, n_test_read=args.n_test_read, 
                                                             batch_size=args.batch_size, n_train_spec_precal=args.n_train_spec_precal,
                                                             input_cond=args.input_cond)

    model = PerformanceNet(input_cond_dim)
    if CUDA_FLAG == 1:
        model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    model.zero_grad()
    optimizer.zero_grad()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
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
    parser.add_argument("--n-train-spec-precal", type=int, default=None, help='How many spectrograms to precalculate')
    parser.add_argument("--input-cond", type=str, default='mel', help='default type for the input conditioning')
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    
    main(args)
