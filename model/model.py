import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
from sklearn.model_selection import train_test_split
import torch.utils.data as utils
import sys
import pickle as pkl

cuda = torch.device("cuda")

def conv1x3(in_channels, out_channels, stride=1, padding=1, bias=True, groups=1):
    return nn.Conv1d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=padding,
        bias=bias,
        groups=groups)

def upconv1x2(in_channels, out_channels, kernel):
    return nn.ConvTranspose1d(
        in_channels,
        out_channels,
        kernel_size=kernel,
        stride=2,
        padding=1
        )


class DownConv(nn.Module):
    def __init__(self, in_channels, out_channels, block_id, pooling = True):
        super(DownConv,self).__init__()
        self.in_channels = in_channels 
        self.out_channels = out_channels 
        self.pooling = pooling            
        self.activation = nn.LeakyReLU(0.01)
        self.conv1 = conv1x3(self.in_channels, self.out_channels) 
        self.conv1_BN = nn.InstanceNorm1d(self.out_channels)
        self.conv2 = conv1x3(self.out_channels, self.out_channels) 
        self.conv2_BN = nn.InstanceNorm1d(self.out_channels)  
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

    def forward(self,x):
        x = self.activation(self.conv1_BN(self.conv1(x)))
        x = self.activation(self.conv1_BN(self.conv2(x)))
        before_pool = x
        if self.pooling:
            x = self.pool(x)
        return x, before_pool


class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels, skip_channels, cond_channels, block_id, activation = nn.LeakyReLU(0.01), upconv_kernel=2):
        super(UpConv, self).__init__()
        self.skip_channels = skip_channels  
        self.in_channels = in_channels 
        self.out_channels = out_channels
        self.cond_channels = cond_channels
        self.activation = activation
        self.upconv = upconv1x2(self.in_channels, self.out_channels,kernel=upconv_kernel)
        self.upconv_BN = nn.InstanceNorm1d(self.out_channels) 
        self.conv1 = conv1x3( self.skip_channels + self.out_channels, self.out_channels)   
        self.conv1_BN = nn.InstanceNorm1d(self.out_channels)
        self.conv2 = conv1x3(self.out_channels + self.cond_channels, self.out_channels) 
        self.conv2_BN = nn.InstanceNorm1d(self.out_channels)
 
    def crop_and_concat(self, upsampled, bypass):
        c = (bypass.size()[2] - upsampled.size()[2]) // 2
        bypass = F.pad(bypass, (-c, -c))
        if bypass.shape[2] > upsampled.shape[2]:
            bypass =  F.pad(bypass, (0, -(bypass.shape[2] - upsampled.shape[2])))  
        else:
            bypass =  F.pad(bypass, ((0, bypass.shape[2] - upsampled.shape[2]) ))
        return torch.cat((upsampled, bypass), 1)

    def forward(self, res, dec, cond):
        x = self.activation(self.upconv_BN(self.upconv(dec)))
        x = self.crop_and_concat(x, res)
        x = self.activation(self.conv1_BN(self.conv1(x)))

        if self.cond_channels:
            x = self.crop_and_concat(x, cond)

        x = self.conv2(x)
        x = self.activation(self.conv2_BN(x))
        return x   


class DenseConcat(nn.Module):
    # allows conditioning on the input audio as well
    # TODO: this only does 1d, maybe extend to 2d...?
    def __init__(self, in_channels, intermediate_channels, out_channels):
        super(DenseConcat, self).__init__()
        self.fc1 = nn.Linear(in_channels, intermediate_channels)
        self.fc2 = nn.Linear(intermediate_channels, out_channels)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, midi_embed, audio_embed):
        x = torch.cat((audio_embed, midi_embed), 1)
        x = x.transpose(1, 2)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = x.transpose(2, 1)
        return x


class Onset_Offset_Encoder(nn.Module):
    def __init__(self, depth = 3, start_channels = 128):
        super(Onset_Offset_Encoder, self).__init__()
        self.start_channels = start_channels
        self.depth = depth
        self.down_convs = [] 
        self.construct_layers()    
        self.down_convs = nn.ModuleList(self.down_convs)
        self.reset_params()
    def construct_layers(self):
        for i in range(self.depth):
            ins = self.start_channels if i == 0 else outs
            outs = self.start_channels * (2 ** (i+1))
            pooling = True if i < self.depth else False
            DC = DownConv(ins, outs, pooling=pooling, block_id = i + 9)
            self.down_convs.append(DC)
    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv1d):
            init.xavier_normal_(m.weight)
            init.constant_(m.bias, 0)
    def reset_params(self):
        for i, m in enumerate(self.modules()):
            self.weight_init(m)
    def forward(self, x):
        condition_tensors = []
        for i, module in enumerate(self.down_convs):
            x,_ = module(x)
            if (i > self.depth - 3):
                condition_tensors.append(x)
        return condition_tensors

class MBRBlock(nn.Module):
    def __init__(self, in_channels, num_of_band):
        super(MBRBlock, self).__init__()
        self.in_dim = in_channels
        self.num_of_band = num_of_band
        self.conv_list1 = []
        self.bn_list1 = []
        self.conv_list2 = []
        self.bn_list2 = []
        self.activation = nn.LeakyReLU(0.01)
        self.band_dim = self.in_dim // self.num_of_band
        for i in range(self.num_of_band):
            self.conv_list1.append(nn.Conv1d(in_channels = self.band_dim, out_channels = self.band_dim, kernel_size = 3, padding = 1))
        for i in range(self.num_of_band):
            self.conv_list2.append(nn.Conv1d(in_channels = self.band_dim, out_channels = self.band_dim, kernel_size = 3, padding = 1))
        for i in range(self.num_of_band):
            self.bn_list1.append(nn.InstanceNorm1d(self.band_dim))
        for i in range(self.num_of_band):  
            self.bn_list2.append(nn.InstanceNorm1d(self.band_dim))  
        self.conv_list1 = nn.ModuleList(self.conv_list1)
        self.conv_list2 = nn.ModuleList(self.conv_list2)        
        self.bn_list1 = nn.ModuleList(self.bn_list1)
        self.bn_list2 = nn.ModuleList(self.bn_list2)

    def forward(self,x):
        bands = torch.chunk(x, self.num_of_band, dim = 1)
        for i in range(len(bands)):
            t = self.activation(self.bn_list1[i](self.conv_list1[i](bands[i])))
            t = self.bn_list2[i](self.conv_list2[i](t))
            torch.add(bands[i],1,t)
        x = torch.add(x,1,torch.cat(bands, dim = 1))
        return x 


class PerformanceNet(nn.Module):
    def __init__(self, depth=5, start_channels=128, start_audio_channels=1025):
        super(PerformanceNet, self).__init__()
        self.depth = depth
        self.start_channels = start_channels 
        self.start_audio_channels = start_audio_channels
        self.construct_layers()
        self.reset_params()
        
    #@staticmethod  
    def construct_layers(self):
        # down convs
        outs_channel_list_midi = []
        self.down_convs = []
        for i in range(self.depth):
            ins = self.start_channels if i == 0 else outs
            outs = self.start_channels * (2 ** (i+1))
            outs_channel_list_midi.append(outs)     # keep track of outs
            pooling = True if i < self.depth-1 else False
            DC = DownConv(ins, outs, pooling=pooling, block_id=i)
            self.down_convs.append(DC)  
        self.down_convs = nn.ModuleList(self.down_convs)
        
        # down convs audio
        outs_channel_list_audio = [
            int(1024*1.5), 2048, int(2048*1.5), 4096, int(4096*1.5)
        ]
        self.down_convs_audio = []
        for i in range(self.depth):
            ins = self.start_audio_channels if i == 0 else outs
            outs = outs_channel_list_audio[i]
            #outs = min(self.start_audio_channels * (2 ** (i+1)), 4096)
            pooling = True if i < self.depth-1 else False
            DC = DownConv(ins, outs, pooling=pooling, block_id=i)
            self.down_convs_audio.append(DC)  
        self.down_convs_audio = nn.ModuleList(self.down_convs_audio)

        # dense layers 
        # in_channels, intermediate_channels, out_channels
        self.dense_concats = []
        for i in range(self.depth):
            out_midi = outs_channel_list_midi[-(i+1)]
            out_audio = outs_channel_list_audio[-(i+1)]
            self.dense_concats.append(DenseConcat(out_midi + out_audio, int(out_midi * 1.5), out_midi)) 
        # self.dense_concats.append(DenseConcat(4096 * 2, int(4096 * 1.5), 4096)) 
        # self.dense_concats.append(DenseConcat(2048 * 2, int(2048 * 1.5), 2048))
        # self.dense_concats.append(DenseConcat(1024 * 2, int(1024 * 1.5), 1024))
        # self.dense_concats.append(DenseConcat(512 * 2, int(512 * 1.5), 512))
        self.dense_concats = nn.ModuleList(self.dense_concats)

        # up convs
        self.up_convs = []
        self.up_convs.append(UpConv(4096,2048,2048, 1024, block_id = 5, upconv_kernel=6))
        self.up_convs.append(UpConv(2048,1024,1024, 512, block_id = 6, upconv_kernel=4))
        self.up_convs.append(UpConv(1024,1024,512,0, block_id= 7, upconv_kernel=3))
        self.up_convs.append(UpConv(1024,1024,256,0, block_id = 8))
        self.up_convs = nn.ModuleList(self.up_convs)

        # multi-band residual blocks
        self.MBRBlock1 = MBRBlock(1024,2) 
        self.MBRBlock2 = MBRBlock(1024,4)
        self.MBRBlock3 = MBRBlock(1024,8)
        self.MBRBlock4 = MBRBlock(1024,16)
        
        # final layers
        self.lastconv = nn.ConvTranspose1d(1024, 1025, kernel_size=3, stride=1, padding=1)
        self.lrelu = nn.LeakyReLU(0.01)

        # onset/offset decoder
        self.onset_offset_encoder = Onset_Offset_Encoder()
        
    @staticmethod  
    def weight_init(m):
        if isinstance(m, nn.Conv1d):
            init.xavier_normal_(m.weight)
            init.constant_(m.bias, 0)
        if isinstance(m, nn.ConvTranspose1d):
            init.xavier_normal_(m.weight)
            init.constant_(m.bias, 0)


    def reset_params(self):
        for i, m in enumerate(self.modules()):
            self.weight_init(m)
    
    def forward(self, x_midi, x_audio, cond):
        # midis 
        encoder_layer_outputs_midi = []
        for i, module in enumerate(self.down_convs):
            x_midi, before_pool = module(x_midi)
            encoder_layer_outputs_midi.append(before_pool)

        # audio spectrograms - I believe they are mel-spectrograms - standard convnets
        # TODO: finish the unet architecture where you save and merge the audio spectrogram
        encoder_layer_outputs_audio = []
        for i, module in enumerate(self.down_convs_audio):
            x_audio, before_pool_audio = module(x_audio)
            encoder_layer_outputs_audio.append(before_pool_audio)

        # concat with dense layers - x output is same as x_midi
        x = self.dense_concats[0](x_midi, x_audio)

        Onoff_Conditions = self.onset_offset_encoder(cond)

        # deconv
        for i, module in enumerate(self.up_convs):
            # get skip-connections from the earlier part of the U-net, merge

            before_pool_midi = encoder_layer_outputs_midi[-(i+2)]
            before_pool_audio = encoder_layer_outputs_audio[-(i+2)]
            before_pool = self.dense_concats[i+1](before_pool_midi, before_pool_audio)

            if i < self.onset_offset_encoder.depth - 1:
                x = module(before_pool, x, Onoff_Conditions[i-1])            
            else:
                x = module(before_pool, x, None)

        # multi-band residual blocks
        x = self.MBRBlock1(x)
        x = self.MBRBlock2(x)
        x = self.MBRBlock3(x)
        x = self.MBRBlock4(x)
        x = self.lrelu(self.lastconv(x)) 
        return x

