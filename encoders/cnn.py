import numpy as np
import torch
import torch.nn as nn
import math

def conv3x3(in_channels=3, out_channels=3, stride=1):
    return torch.nn.Conv2d(
        in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
    )

def calculate_conv_dims(h_w, kernel_size=3, stride=1, padding=0, dilation=1):
    """
    (H −1)×stride[0]−2×padding[0]+dilation[0]×(kernel_size[0]−1)+output_padding[0]+1
W_{o
    """
    h_out = h_w[0]*stride - 2*padding + dilation * ( kernel_size - 1 ) + padding
    w_out = h_w[1]*stride - 2*padding + dilation * ( kernel_size - 1 ) + padding
    return (h_out, w_out)

def calculate_pool_dims(h_w, kernel_size=3, stride=1, padding=0, dilation=1):
    """
    (H −1)×stride[0]−2×padding[0]+dilation[0]×(kernel_size[0]−1)+output_padding[0]+1
W_{o
    """
    h_out = h_w[0]*stride - 2*padding + dilation * ( kernel_size - 1 ) + padding
    w_out = h_w[1]*stride - 2*padding + dilation * ( kernel_size - 1 ) + padding
    return (h_out, w_out)

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
        mid_kernel_size = 6

#         l1_dims = calculate_conv_dims(h_w, kernel_size=mid_kernel_size, stride=2, padding=2)
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(input_channels, mid_channels, kernel_size=mid_kernel_size, stride=2, padding=2),
            torch.nn.LeakyReLU(),
#             torch.nn.MaxPool2d(kernel_size=mid_kernel_size, stride=1, return_indices=True)
        )

#         l2_dims = calculate_conv_dims(l1_dims, kernel_size=mid_kernel_size, stride=2, padding=2)
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(mid_channels, mid_channels, kernel_size=mid_kernel_size, stride=2, padding=2),
            torch.nn.LeakyReLU(),
#             torch.nn.MaxPool2d(kernel_size=mid_kernel_size, stride=1, return_indices=True)
        )

#         l3_dims = calculate_conv_dims(l2_dims, kernel_size=mid_kernel_size, stride=2, padding=2)
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(mid_channels, latent_channels, kernel_size=mid_kernel_size, stride=2, padding=2),
            torch.nn.LeakyReLU(),
#             torch.nn.MaxPool2d(kernel_size=mid_kernel_size, stride=1, return_indices=True)
        )
        
        self.encoder = torch.nn.ModuleList([self.conv1, self.conv2, self.conv3])
        
#         self.unpool1 = torch.nn.MaxUnpool2d(kernel_size=mid_kernel_size, stride=1)
        self.deconv1 = torch.nn.ConvTranspose2d(latent_channels, mid_channels, kernel_size=mid_kernel_size, stride=2, padding=2)
        self.decode_activ1 = torch.nn.ReLU()
        
#         self.unpool2 = torch.nn.MaxUnpool2d(kernel_size=mid_kernel_size, stride=1)
        self.deconv2 = torch.nn.ConvTranspose2d(mid_channels, mid_channels, kernel_size=mid_kernel_size, stride=2, padding=2)
        self.decode_activ2 = torch.nn.ReLU()
        
#         self.unpool3 = torch.nn.MaxUnpool2d(kernel_size=mid_kernel_size, stride=1)
        self.deconv3 = torch.nn.ConvTranspose2d(mid_channels, input_channels, kernel_size=mid_kernel_size, stride=2, padding=2)
        self.decode_activ3 = torch.nn.ReLU()
        self.batchnorm = torch.nn.BatchNorm2d(input_channels)

#         self.decoder = torch.nn.ModuleList([self.deconv1, self.deconv2, self.deconv3])
                
#         self.resblocks = torch.nn.Sequential(*[ResidualBlock(output_channels) for _ in range(num_blocks)])

    def forward(self, x):
#         pool_indices = []
        for layer in self.encoder:
            x = layer(x)
#             print(x.shape, layer)
#             pool_indices.append(indices)

        self.latent_vector = x

#         pool_indices.reverse()
#         x = self.unpool1(x, pool_indices[0])
        x = self.deconv1(x)
        x = self.decode_activ1(x)
        
#         print(x.shape, pool_indices[1].shape)
#         x = self.unpool2(x, pool_indices[1])
        x = self.deconv2(x)
        x = self.decode_activ2(x)
        x = self.batchnorm(x)
        
#         x = self.unpool3(x, pool_indices[2])
        x = self.deconv3(x)
        x = self.decode_activ3(x)
        x = torch.clamp(x, max=255)
        return x
