import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
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
    """
    TODO: fixme
    """
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

class AutoEncoder(pl.LightningModule):
    def __init__(
        self,
        input_channels,
        h_w=(400,400),
        num_blocks=2,
        mid_channels=10,
    ):
        super().__init__()
        self.learning_rate = 1e-3
        self.batch_size = 100

        conv0 = torch.nn.Sequential(
            torch.nn.Conv2d(input_channels, mid_channels, kernel_size=6, stride=2, padding=0),
            torch.nn.BatchNorm2d(mid_channels),
            torch.nn.ReLU()
        )

        conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(mid_channels, mid_channels, kernel_size=5, stride=2, padding=0),
            torch.nn.BatchNorm2d(mid_channels),
            torch.nn.ReLU()
        )

        conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(mid_channels, input_channels, kernel_size=5, stride=1, padding=0),
            torch.nn.BatchNorm2d(input_channels),
            torch.nn.ReLU()
        )

        max_pool = nn.MaxPool2d(2, stride=2, return_indices=True)

        self.encoder = torch.nn.ModuleList([
            conv0,
            conv1,
            conv2,
#             max_pool
        ])

        deconv0 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(3, mid_channels, kernel_size=5, stride=1, padding=0),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(mid_channels)
        )

        deconv1 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(mid_channels, mid_channels, kernel_size=5, stride=2, padding=0),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(mid_channels)
        )

        deconv2 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(mid_channels, input_channels, kernel_size=8, stride=2, padding=0),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(input_channels),
        )

        unpool = nn.MaxUnpool2d(2, stride=2)

        self.decoder = torch.nn.ModuleList([
#             unpool,
            deconv0,
            deconv1,
            deconv2
        ])

    def forward(self, x):
        pool_indices = []
        for layer in self.encoder:
            if type(layer) == torch.nn.modules.pooling.MaxPool2d:
                size = x.size()
                x, indices = layer(x)
                pool_indices += [[indices, size]] # insert at beginning so iterating is more convenient
            else:
                x = layer(x)

        self.latent_vector = x

        pool_index = 0
        for layer in self.decoder:
            if type(layer) == torch.nn.modules.pooling.MaxUnpool2d:
                indices, output_size = pool_indices[pool_index]
                x = layer(x, indices, output_size=output_size)
                pool_index += 1
            else:
                x = layer(x)

        x = torch.clamp(x, 0, 1)
        return x

    def training_step(self, batch, batch_idx):
        img = batch
        recon = self.forward(img)
        loss = F.mse_loss(recon, img)

        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

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
