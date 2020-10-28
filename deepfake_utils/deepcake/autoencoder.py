import torch
import torch.nn as nn
import torch.utils.data
from torch import nn, optim
from .conv_nd import Conv2d
from .autoencoder_utils import _ConvLayer, _UpScale, _PixelShuffler, Flatten, Reshape



class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()

        self.encoder = nn.Sequential(
            _ConvLayer(3, 128),
            _ConvLayer(128, 256),
            _ConvLayer(256, 512),
            _ConvLayer(512, 1024),
            Flatten(),
            nn.Linear(1024 * 4 * 4 , 1024),    ## extra *4 for 128 images

            nn.Linear(1024, 1024 * 4 * 4),
            Reshape(),
            _UpScale(1024, 512),
        )

        self.decoder_A = nn.Sequential(
            _UpScale(512, 256),
            _UpScale(256, 128),
            _UpScale(128, 64),

            # _UpScale(64, 64),  ## extra for 128, 128 image 

            Conv2d(64, 3, kernel_size=5, padding=1),
            nn.Sigmoid(),
        )

        self.decoder_B = nn.Sequential(
            _UpScale(512, 256),
            _UpScale(256, 128),
            _UpScale(128, 64),

            # _UpScale(64, 64),  ## extra for 128, 128 image 


            Conv2d(64, 3, kernel_size=5, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x,  decoder ='A'):
        if decoder == 'A':
            out = self.encoder(x)
            out = self.decoder_A(out)
        else:
            out = self.encoder(x)
            out = self.decoder_B(out)
        return out

class Autoencoder_with_landmarks(nn.Module):
    def __init__(self):
        super(Autoencoder_with_landmarks, self).__init__()

        self.encoder = nn.Sequential(
            _ConvLayer(3, 128),
            _ConvLayer(128, 256),
            _ConvLayer(256, 512),
            _ConvLayer(512, 1024),
            Flatten(),
            nn.Linear(1024 * 4 * 4 , 1024),    ## extra *4 for 128 images
            nn.Linear(1024, (1024 * 4 * 4) - 68*2 ),  ## - 68*2 for landmarks accomodation without messsing with the sizes 
            
        )

        self.decoder_A = nn.Sequential(

            Reshape(),
            _UpScale(1024, 512),
            _UpScale(512, 256),
            _UpScale(256, 128),
            _UpScale(128, 64),

            # _UpScale(64, 64),  ## extra for 128, 128 image 

            Conv2d(64, 3, kernel_size=5, padding=1),
            nn.Sigmoid(),
        )

        self.decoder_B = nn.Sequential(

            Reshape(),
            _UpScale(1024, 512),
            _UpScale(512, 256),
            _UpScale(256, 128),
            _UpScale(128, 64),

            # _UpScale(64, 64),  ## extra for 128, 128 image 


            Conv2d(64, 3, kernel_size=5, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, pack,  decoder ='A'):
        x = pack["x"]
        l_x = pack["l_x"]
        l_y = pack["l_y"]

        if decoder == 'A':
            out = self.encoder(x)
            out_cat = torch.cat([out, l_x, l_y], dim = 1)
            out = self.decoder_A(out_cat)
        else:
            out = self.encoder(x)
            out_cat = torch.cat([out, l_x, l_y], dim = 1)

            out = self.decoder_B(out_cat)
        return out
