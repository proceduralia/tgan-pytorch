import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import os
from torch.autograd import Variable

class FrameSeedGenerator(nn.Module):
    #Generate exactly 16 latent vectors starting from 1
    def __init__(self, z_slow_dim, z_fast_dim):
        super().__init__()
        self.z_slow_dim = z_slow_dim
        self.z_fast_dim = z_fast_dim

        self.dc0 = nn.ConvTranspose1d(z_slow_dim, 512, 1, 1, 0)
        self.dc1 = nn.ConvTranspose1d(512, 256, 4, 2, 1) 
        self.dc2 = nn.ConvTranspose1d(256, 128, 4, 2, 1)
        self.dc3 = nn.ConvTranspose1d(128, 128, 4, 2, 1)
        self.dc4 = nn.ConvTranspose1d(128, z_fast_dim, 4, 2, 1)
        self.bn0 = nn.BatchNorm1d(512)
        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(128)
    
    def forward(self, z_slow):
        h = z_slow.view(z_slow.size(0),-1, 1)
        h = F.relu(self.bn0(self.dc0(h)))
        h = F.relu(self.bn1(self.dc1(h)))
        h = F.relu(self.bn2(self.dc2(h)))
        h = F.relu(self.bn3(self.dc3(h)))
        z_fast = F.tanh(self.dc4(h))
        return z_fast

class VideoGenerator(nn.Module):
    def __init__(self, z_slow_dim, z_fast_dim, out_channels=3, bottom_width=4, conv_ch=512):
        super().__init__()
        self.ch = conv_ch
        self.bottom_width = bottom_width
        slow_mid_dim = bottom_width * bottom_width * conv_ch //2
        fast_mid_dim = bottom_width * bottom_width * conv_ch //2
        
        self.l0s = nn.Linear(z_slow_dim, slow_mid_dim)
        self.l0f = nn.Linear(z_fast_dim, fast_mid_dim)
        self.dc1 = nn.ConvTranspose2d(conv_ch, conv_ch // 2, 4, 2, 1)
        self.dc2 = nn.ConvTranspose2d(conv_ch // 2, conv_ch // 4, 4, 2, 1)
        self.dc3 = nn.ConvTranspose2d(conv_ch // 4, conv_ch // 8, 4, 2, 1)
        self.dc4 = nn.ConvTranspose2d(conv_ch // 8, conv_ch // 16, 4, 2, 1)
        self.dc5 = nn.ConvTranspose2d(conv_ch // 16, out_channels, 3, 1, 1)
        
        self.bn0s = nn.BatchNorm1d(slow_mid_dim)
        self.bn0f = nn.BatchNorm1d(fast_mid_dim)
        self.bn1 = nn.BatchNorm2d(conv_ch // 2)
        self.bn2 = nn.BatchNorm2d(conv_ch // 4)
        self.bn3 = nn.BatchNorm2d(conv_ch // 8)
        self.bn4 = nn.BatchNorm2d(conv_ch // 16)
    
    def forward(self, z_slow, z_fast):
        n = z_slow.size(0)
        h_slow = (F.relu(self.bn0s(self.l0s(z_slow))).view(n, self.ch // 2, self.bottom_width, self.bottom_width))
        h_fast = (F.relu(self.bn0f(self.l0f(z_fast)))).view(n, self.ch // 2, self.bottom_width, self.bottom_width)
        h = torch.cat((h_slow, h_fast), 1)
        h = F.relu(self.bn1(self.dc1(h)))
        h = F.relu(self.bn2(self.dc2(h)))
        h = F.relu(self.bn3(self.dc3(h)))
        h = F.relu(self.bn4(self.dc4(h)))
        x = F.tanh(self.dc5(h))
        return x

class Model(nn.Module):
    def __init__(self, z_slow_dim=256, z_fast_dim=256, out_channels=3, bottom_width=4, conv_ch=512):
        super().__init__()
        self.z_slow_dim = z_slow_dim
        self.z_fast_dim = z_fast_dim
        self.out_channels = out_channels
        self._fsgen = FrameSeedGenerator(z_slow_dim, z_fast_dim)
        self._vgen = VideoGenerator(z_slow_dim,z_fast_dim, out_channels, bottom_width,conv_ch)

    def generate_input(self, batch_size=16):
        """
        Generates latent vector from normal distribution
        """
        z_slow = torch.randn(batch_size, self.z_slow_dim)
        return z_slow

    def forward(self, z_slow):
        z_fast = self._fsgen(z_slow)
        B, n_z_fast, n_frames = z_fast.size()
        z_fast = z_fast.permute(0, 2, 1).contiguous().view(B * n_frames, n_z_fast) #squash time dimension in batch dimension
        
        B, n_z_slow = z_slow.size()
        z_slow = z_slow.unsqueeze(1).repeat(1, n_frames, 1)
        z_slow = z_slow.contiguous().view(B * n_frames, n_z_slow)
        
        out = self._vgen(z_slow, z_fast)
        out = out.view(B, n_frames, self.out_channels, 64, 64)
        return out

if __name__ == "__main__":
    #The number of frames in a video is fixed at 16
    batch_size = 8 
    gen = Model()
    z_slow = Variable(gen.generate_input(batch_size)) 
    out = gen(z_slow)
    print("Output video generator:", out.size())
