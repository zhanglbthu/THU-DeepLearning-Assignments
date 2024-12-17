import torch
import torch.nn as nn
import torch.nn.functional as F


def zero_module(module):
    for p in module.parameters():
        p.detach().zero_()
    return module


def Normalize(in_channels, num_groups=8):
    return torch.nn.GroupNorm(
        num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True
    )


class Upsample(nn.Module):
    def forward(self, x, *args, **kwargs):
        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        return x


class Downsample(nn.Module):
    def forward(self, x, *args, **kwargs):
        x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x


class ResnetBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels=None,
        emb_channels=0,
    ):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels

        self.norm1 = Normalize(in_channels)
        self.conv1 = torch.nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        if emb_channels > 0:
            self.emb_proj = torch.nn.Linear(emb_channels, out_channels)
        self.norm2 = Normalize(out_channels)
        self.conv2 = torch.nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        if self.in_channels != self.out_channels:
            self.nin_shortcut = torch.nn.Conv2d(
                in_channels, out_channels, kernel_size=1, stride=1, padding=0
            )

    def forward(self, x, emb=None):
        h = x
        h = self.norm1(h)
        h = F.silu(h)
        h = self.conv1(h)

        if emb is not None:
            h = h + self.emb_proj(F.silu(emb))[:, :, None, None]

        h = self.norm2(h)
        h = F.silu(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            x = self.nin_shortcut(x)

        return x + h


def make_down_block(in_channels, out_channels, emb_channels=0):
    return nn.ModuleList([
        ResnetBlock(in_channels, out_channels, emb_channels),
        ResnetBlock(out_channels, out_channels, emb_channels),
        Downsample(),
    ])

def make_up_block(in_channels, out_channels, emb_channels=0):
    return nn.ModuleList([
        Upsample(),
        ResnetBlock(in_channels, out_channels, emb_channels),
        ResnetBlock(out_channels, out_channels, emb_channels),
    ])



class Encoder(nn.Module):
    def __init__(
        self,
        in_channels=3,
        z_channels=3,
        num_channels=32,
    ):
        super().__init__()
        self.num_channels = num_channels
        self.in_channels = in_channels

        self.conv_in = nn.Conv2d(
            in_channels, self.num_channels, kernel_size=3, stride=1, padding=1
        )

        self.down_blocks = nn.ModuleList([
            make_down_block(self.num_channels, self.num_channels),
            make_down_block(self.num_channels, 2 * self.num_channels),
        ])

        self.mid_block = nn.ModuleList([
            ResnetBlock(2 * self.num_channels, 4 * self.num_channels),
            ResnetBlock(4 * self.num_channels, 4 * self.num_channels),
        ])

        self.conv_out = nn.Sequential(
            Normalize(4 * self.num_channels),
            nn.SiLU(),
            nn.Conv2d(
                4 * self.num_channels, z_channels, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        h = self.conv_in(x)
        for down_block in self.down_blocks:
            for down_module in down_block:
                h = down_module(h)
        for mid_module in self.mid_block:
            h = mid_module(h)
        h = self.conv_out(h)
        return h


class Decoder(nn.Module):
    def __init__(
        self,
        z_channels=3,
        out_channels=3,
        num_channels=32,
    ):
        super().__init__()
        self.num_channels = num_channels
        self.out_channels = out_channels

        self.conv_in = nn.Conv2d(
            z_channels, 4 * self.num_channels, kernel_size=3, stride=1, padding=1
        )

        self.mid_block = nn.ModuleList([
            ResnetBlock(4 * self.num_channels, 4 * self.num_channels),
            ResnetBlock(4 * self.num_channels, 2 * self.num_channels),
        ])
        
        self.up_blocks = nn.ModuleList([
            make_up_block(2 * self.num_channels, self.num_channels),
            make_up_block(self.num_channels, self.num_channels),
        ])

        self.conv_out = nn.Sequential(
            Normalize(self.num_channels),
            nn.SiLU(),
            nn.Conv2d(
                self.num_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        h = self.conv_in(x)
        for mid_module in self.mid_block:
            h = mid_module(h)
        for up_block in self.up_blocks:
            for up_module in up_block:
                h = up_module(h)
        h = self.conv_out(h)
        return h
