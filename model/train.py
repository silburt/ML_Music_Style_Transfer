import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
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

CUDA_FLAG = 0
if torch.cuda.is_available():
    cuda = torch.device("cuda")
    CUDA_FLAG = 1

DEFAULT_STYLES = ['cuba', 'aliciakeys', 'gentleman', 'harpsichord', 'markisuitcase', 'upright']

class hyperparams(object):
    def __init__(self, args):
        self.instrument = args.instrument
        self.train_epoch = args.epochs
        self.test_freq = args.test_freq
        self.exp_name = args.exp_name
        self.iter_train_loss = []
        self.iter_test_loss = []
        self.loss_history = []
        self.test_loss_history = []
        self.best_loss = 1e10 
        self.best_epoch = 0


class Dataseth5py(torch.utils.data.Dataset):
    # https://discuss.pytorch.org/t/how-to-speed-up-the-data-loader/13740/3
    def __init__(self, in_file, seed=42, n_read=None):
        super(Dataseth5py, self).__init__()

        self.dataset = h5py.File(in_file, 'r')
        self.styles = [name for name in self.dataset.keys() if 'spec_' in name] # get styles from the data

        # TODO: the big issue is you need to optimize how to read data into memory from h5py
        # loading one-by-one is way too slow (a few seconds vs. microseconds). Note that the 
        # rest of the profiling times are: concat/transpose ~ 0.005s, FloatTensor ~ 0.02 
        # (and thus, after this loading issue is solved FloatTensor becomes the bottleneck unless it 
        # can be moved to the main train() function and be applied to batches vs individual items here)
        if n_read is not None:
            self.pianoroll = self.dataset['pianoroll'][:n_read]
            self.onoff = self.dataset['onoff'][:n_read]
            self.specs = {}
            for style in self.styles:
                self.specs[style] = self.dataset[f'spec_{style}'][:n_read] 
        else:
            self.pianoroll = self.dataset['pianoroll'][:]
            self.onoff = self.dataset['onoff'][:]
            self.specs = {}
            for style in self.styles:
                self.specs[style] = self.dataset[f'spec_{style}'][:]

        self.n_data = self.pianoroll.shape[0]
        random.seed(seed)

    def __getitem__(self, index):
        '''
        The input data are the pianoroll, onoff, a *random* spec from the same style
        The output data is the *matching* spec for the corresponding pianoroll/onoff
        '''
        # piano
        pianoroll = self.pianoroll[index]
        onoff = self.onoff[index]
        pianoroll = np.concatenate((pianoroll, onoff), axis = -1)
        pianoroll = np.transpose(pianoroll, (1, 0))

        # specs
        style = random.choice(self.styles)
        spec = self.specs[style][index]
        spec_rand = self.specs[style][random.randint(0, self.n_data)]

        if CUDA_FLAG == 1:
            X = torch.cuda.FloatTensor(pianoroll)
            X_cond = torch.cuda.FloatTensor(spec_rand)
            y = torch.cuda.FloatTensor(spec)
        else:
            X = torch.Tensor(pianoroll)
            X_cond = torch.Tensor(spec_rand)
            y = torch.Tensor(spec)
        return X, X_cond, y

    def __len__(self):
        return self.n_data


def Process_Data(data_dir, data_basename, n_train_read=None, batch_size=16):
    train_dataset = Dataseth5py(os.path.join(data_dir, data_basename + '_train.h5py'), n_read=n_train_read)
    test_dataset = Dataseth5py(os.path.join(data_dir, data_basename + '_test.h5py'))

    kwargs = {}
    train_loader = utils.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
    test_loader = utils.DataLoader(test_dataset, batch_size=batch_size, **kwargs)
    return train_loader, test_loader


def train(model, epoch, train_loader, optimizer, iter_train_loss):
    model.train()
    train_loss = 0
    for batch_idx, (data, data_cond, target) in enumerate(train_loader):        
        optimizer.zero_grad()
        split = torch.split(data, 128, dim=1)
        loss_function = nn.MSELoss()
        if CUDA_FLAG == 1:
            y_pred = model(split[0].cuda(), data_cond.cuda(), split[1].cuda())
            loss = loss_function(y_pred, target.cuda())
        else:
            y_pred = model(split[0], data_cond, split[1]) 
            loss = loss_function(y_pred, target)
        
        loss.backward()
        iter_train_loss.append(loss.item())
        train_loss += loss
        optimizer.step()    
         
        if batch_idx % 2 == 0:
            print ('Train Epoch: {} [{}/{} ({:.0f}%)]\t Loss: {:.6f}'.format(epoch, batch_idx * len(data), len(train_loader.dataset), 100. * batch_idx/len(train_loader), loss.item()/len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss/ len(train_loader.dataset)))
    return train_loss / len(train_loader.dataset)


def test(model, epoch, test_loader, scheduler, iter_test_loss):
    with torch.no_grad():
        model.eval()
        test_loss = 0
        for idx, (data, data_cond, target) in enumerate(test_loader):
            split = torch.split(data, 128, dim=1)
            loss_function = nn.MSELoss()
            if CUDA_FLAG == 1:
                y_pred = model(split[0].cuda(), data_cond.cuda(), split[1].cuda())
                loss = loss_function(y_pred, target.cuda())
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
    train_loader, test_loader = Process_Data(args.data_dir, args.dataset_basename, n_train_read=args.n_train_read, 
                                             batch_size=args.batch_size)
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
    parser.add_argument("-dataset-basename", type=str, default='style_transfer', help="basename of train/test data h5py file")
    parser.add_argument("-epochs", type=int, default=1)
    parser.add_argument("-test-freq", type=int, default=1, help='how many epochs between running against test data')
    parser.add_argument("-exp-name", type=str, default='piano_test')
    parser.add_argument("--n-train-read", type=int, default=None, help='How many data points to read (length of an epoch)')
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    
    main(args)
