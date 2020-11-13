import numpy as np
import torch
import torch.nn as nn
import math

def conv3x3(in_channels=3, out_channels=3, stride=1):
    return torch.nn.Conv2d(
        in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
    )

def isfloat(value):
    try:
        float(value)
        return True
    except ValueError:
        return False

def get_layer(str):
    name, attrs = str.split('(')
    attrs = attrs.replace(')', '')
    attrs = attrs.split(', ')
    kwargs = {}
    for attr in attrs:
        if attr == '' or attr == None:
            pass
        else:
            k,v = attr.split('=')
            if v.isdigit():
                v = int(v)
            elif isfloat(v):
                v = float(v)
            elif v in ['True', 'False']:
                v = bool(v)
            kwargs[k] = v

    mod = __import__('torch.nn', fromlist=[name])
    klass = getattr(mod, name)
    return klass(**kwargs)

class ResidualBlock(torch.nn.Module):
    def __init__(self, num_channels, stride=1):
        super().__init__()
        self.conv1 = conv3x3(num_channels, num_channels, stride, return_indices=True)
        self.conv1_indices = []
        self.bn1 = torch.nn.BatchNorm2d(num_channels)
        self.conv2_indices = []
        self.conv2 = conv3x3(num_channels, num_channels, return_indices=True)
        self.bn2 = torch.nn.BatchNorm2d(num_channels)

    def forward(self, x):
        x, self.conv1_indices = self.conv1(x)
        x = self.bn1(x)
        x = torch.nn.functional.relu(x)
        x, self.conv2_indices = self.conv2(x)
        x = self.bn2(x)
        x += x
        x = torch.nn.functional.relu(x)
        return x

class AutoEncoder(torch.nn.Module):
    def __init__(
        self,
        input_channels,
        h_w=(400,400),
        num_blocks=2,
        latent_channels=1,
    ):
        super().__init__()
        mid_channels = input_channels
        mid_kernel_size = 4

        self.conv0 = torch.nn.Sequential(
            torch.nn.Conv2d(input_channels, mid_channels, kernel_size=5, stride=1, padding=1),
            torch.nn.BatchNorm2d(3),
            torch.nn.ReLU()
        )

        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(input_channels, mid_channels, kernel_size=mid_kernel_size, stride=2, padding=2),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(mid_channels)
        )

        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(mid_channels, mid_channels, kernel_size=mid_kernel_size, stride=2, padding=2),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(mid_channels)
        )

        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(mid_channels, latent_channels, kernel_size=mid_kernel_size, stride=2, padding=2),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(mid_channels)
        )

        self.encoder = torch.nn.ModuleList([self.conv0, self.conv1, self.conv2, self.conv3])

        self.deconv1 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(mid_channels, mid_channels, kernel_size=mid_kernel_size, stride=2, padding=2),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(mid_channels)
        )

        self.deconv2 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(mid_channels, mid_channels, kernel_size=mid_kernel_size, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(mid_channels)
        )

        self.deconv3 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(mid_channels, input_channels, kernel_size=mid_kernel_size, stride=2, padding=1),
            torch.nn.Sigmoid(),
        )

        self.deconv4 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(mid_channels, input_channels, kernel_size=5, stride=1, padding=2),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(3)
        )

        self.decoder = torch.nn.ModuleList([self.deconv1, self.deconv2, self.deconv3, self.deconv4])

#         self.resblocks = torch.nn.Sequential(*[ResidualBlock(output_channels) for _ in range(num_blocks)])

    def forward(self, x):
        for layer in self.encoder:
            x = layer(x)

        self.latent_vector = x

        for layer in self.decoder:
            x = layer(x)

        return x

    def save(self, filename, optimizer=None, loss=None, epoch=-1):
        encoder_layers = [l.__repr__() for l in self.encoder.children()]
        decoder_layers = [l.__repr__() for l in self.decoder.children()]
        state = {
            'encoder': encoder_layers,
            'decoder': decoder_layers,
            'model_state_dict': self.state_dict(),
            'epoch': epoch,
        }

        if optimizer != None:
            state['optimizer_state_dict'] = optimizer.state_dict()

        if loss != None:
            state['loss'] = loss.item()

        torch.save(state, filename)

    def load(self, filename, optimizer=None):
        checkpoint = torch.load(filename)
        self.encoder = nn.Sequential(*[get_layer(l) for l in checkpoint['encoder']])
        self.decoder = nn.Sequential(*[get_layer(l) for l in checkpoint['decoder']])
        self.load_state_dict(checkpoint['model_state_dict'])

        if optimizer != None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
