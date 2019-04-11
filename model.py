import torch
import torch.nn as nn
import torch.nn.functional as F


class Interpolate(nn.Module):
    """
    This exists only because nn.Upsample is deprecated and I need nn.Sequential
    """
    def __init__(self, size=None, scale_factor=None, mode='nearest'):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        x = self.interp(x, size=self.size, scale_factor=self.scale_factor, mode=self.mode)
        return x


class Generator(nn.Module):
    def __init__(self, input_dim=131, output_channel=3):
        super(Generator, self).__init__()
        # Input shape: (N, 131)
        self.FC = nn.Linear(input_dim, 8192)
        # Reshape to: (128 x 8 x 8)
        self.main = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),  # 128 x 8 x 8
            nn.ELU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),  # 128 x 8 x 8
            nn.ELU(),
            Interpolate(scale_factor=(2, 2), mode='nearest'),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),  # 128 x 16 x 16
            nn.ELU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),  # 128 x 16 x 16
            nn.ELU(),
            Interpolate(scale_factor=(2, 2), mode='nearest'),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),  # 128 x 32 x 32
            nn.ELU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),  # 128 x 32 x 32
            nn.ELU(),
            Interpolate(scale_factor=(2, 2), mode='nearest'),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),  # 128 x 64 x 64
            nn.ELU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),  # 128 x 64 x 64
            nn.ELU(),
            nn.Conv2d(128, output_channel, kernel_size=3, stride=1, padding=1)  # output_channel x 64 x 64
        )

    def forward(self, x):
        x = self.FC(x)
        x = x.view(-1, 128, 8, 8)
        x = self.main(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, config):
        super(Discriminator, self).__init__()
        # Input shape: (N, 6, 64, 64)
        self.encoder = nn.Sequential(
            nn.Conv2d(6, 128, kernel_size=3, stride=1, padding=1),    # 128 x 64 x 64
            nn.ELU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),  # 128 x 64 x 64
            nn.ELU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),  # 128 x 64 x 64
            nn.ELU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),  # 128 x 32 x 32
            nn.ELU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),  # 128 x 32 x 32
            nn.ELU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),  # 128 x 32 x 32
            nn.ELU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),  # 128 x 16 x 16
            nn.ELU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),  # 128 x 16 x 16
            nn.ELU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),  # 128 x 16 x 16
            nn.ELU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),  # 128 x 8 x 8
            nn.ELU(),
        )
        # Reshape to: (N, 8192)
        self.FC1 = nn.Linear(8192, 128)
        self.decoder = Generator(input_dim=128, output_channel=6)

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(-1, 8192)
        x = self.FC1(x)
        x = self.decoder(x)
        return x
