import math
import torch
import torch.nn as nn
from module import Normalize, ResnetBlock, make_down_block, make_up_block, Encoder, Decoder, zero_module


class VAE(nn.Module):
    def __init__(
        self,
        img_channels=3,
        z_channels=3,
        num_channels=32,
    ):
        super().__init__()
        self.encoder = Encoder(img_channels, 2 * z_channels, num_channels)
        self.decoder = Decoder(z_channels, img_channels, num_channels)

    def forward(self, x):
        mu, logvar = self.encoder(x).chunk(2, dim=1)
        ############################ Your code here ############################
        # TODO: implement the reparameterization trick to obtain latent z
        ########################################################################
        z = None
        ########################################################################
        x_recon = self.decoder(z)
        return x_recon, mu, logvar
    
    def encode(self, x):
        mu, _ = self.encoder(x).chunk(2, dim=1)
        return mu
    
    def decode(self, z):
        return self.decoder(z)


class Discriminator(nn.Module):

    def __init__(self,
                 in_channels=3, 
                 num_channels=32):
        super().__init__()
        sequence = [
            nn.Conv2d(in_channels, num_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, True),
        ]
        sequence += [
            nn.Conv2d(num_channels, 2 * num_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(2 * num_channels),
            nn.LeakyReLU(0.2, True),
        ]
        sequence += [
            nn.Conv2d(2 * num_channels, 2 * num_channels, kernel_size=4, stride=1, padding=1, bias=False),
        ]
        sequence += [
            nn.Conv2d(2 * num_channels, 1, kernel_size=4, stride=1, padding=1),
        ]
        self.main = nn.Sequential(*sequence)
        self.apply(Discriminator.weights_init)
    
    @staticmethod
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

    def forward(self, x):
        return self.main(x)


class UNet(nn.Module):
    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        num_classes=10,
        num_channels=32,
    ):
        super().__init__()
        self.num_channels = num_channels
        self.time_embed = nn.Sequential(
            nn.Linear(num_channels, num_channels),
            nn.SiLU(),
            nn.Linear(num_channels, num_channels),
        )
        self.label_embed = nn.Sequential(
            nn.Embedding(num_classes, num_channels),
            nn.SiLU(),
            nn.Linear(num_channels, num_channels),
        )
        self.conv_in = nn.Conv2d(in_channels, num_channels, kernel_size=3, stride=1, padding=1)
        self.down_blocks = nn.ModuleList([
            make_down_block(num_channels, 2 * num_channels, num_channels),
            make_down_block(2 * num_channels, 4 * num_channels, num_channels),
        ])
        self.mid_block = nn.ModuleList([
            ResnetBlock(4 * num_channels, 4 * num_channels, num_channels),
            ResnetBlock(4 * num_channels, 4 * num_channels, num_channels),
        ])
        self.up_blocks = nn.ModuleList([
            make_up_block(8 * num_channels, 2 * num_channels, num_channels),
            make_up_block(4 * num_channels, num_channels, num_channels),
        ])
        self.conv_out = nn.Sequential(
            Normalize(num_channels),
            nn.SiLU(),
            zero_module(
                nn.Conv2d(num_channels, out_channels, kernel_size=3, stride=1, padding=1)
            )
        )
    
    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, x, y, t):
        t = self.timestep_embedding(t, self.num_channels)
        t = self.time_embed(t)
        y = self.label_embed(y)
        emb = t + y
        h = self.conv_in(x)
        hs = []
        for down_block in self.down_blocks:
            for down_module in down_block:
                h = down_module(h, emb)
            hs.append(h)
        for mid_module in self.mid_block:
            h = mid_module(h, emb)
        for up_block, h_skip in zip(self.up_blocks, reversed(hs)):
            h = torch.cat([h, h_skip], dim=1)
            for up_module in up_block:
                h = up_module(h, emb)
        h = self.conv_out(h)
        return h
