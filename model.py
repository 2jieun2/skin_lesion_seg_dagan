import torch
from torch import nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, dropout=False, pad=1, d_rate=1):
        super().__init__()
        layers = [
            nn.Conv2d(in_c, out_c, kernel_size=3, stride=1, padding=pad, dilation=d_rate, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))
        layers.extend([
            nn.Conv2d(out_c, out_c, kernel_size=3, stride=1, padding=pad, dilation=d_rate, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        ])
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        out = self.model(x)
        return out


class UpMerge(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=1, stride=1, padding=0, bias=False)
        )
        self.conv = ConvBlock(out_c*2, out_c)

    def forward(self, x, skip_x):
        x = F.upsample(x, size=skip_x.shape[2:], mode='bilinear')
        x = self.conv1x1(x)
        x = torch.cat((x, skip_x), dim=1)
        x = self.conv(x)
        return x


class DenseDilatedBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.pooling = nn.MaxPool2d(kernel_size=2)

        d_rate = [2, 4, 8, 16]

        self.conv1 = ConvBlock(in_c, out_c, pad=d_rate[0], d_rate=d_rate[0])
        self.conv2 = ConvBlock(in_c + out_c, out_c, pad=d_rate[1], d_rate=d_rate[1])
        self.conv3 = ConvBlock(in_c + out_c*2, out_c, pad=d_rate[2], d_rate=d_rate[2])
        self.conv4 = ConvBlock(in_c + out_c*3, out_c, pad=d_rate[3], d_rate=d_rate[3])

        self.out = nn.Conv2d(in_c + out_c*4, out_c, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        d1 = self.conv1(x)

        d2 = torch.cat([x, d1], dim=1)
        d2 = self.conv2(d2)

        d3 = torch.cat([x, d1, d2], dim=1)
        d3 = self.conv3(d3)

        d4 = torch.cat([x, d1, d2, d3], dim=1)
        d4 = self.conv4(d4)

        out = torch.cat([x, d1, d2, d3, d4], dim=1)
        out = self.out(out)

        return out


class GeneratorUNet(nn.Module):
    def __init__(self, args, in_c=3, out_c=1):
        super().__init__()
        self.args = args
        nf = self.args.nf

        self.pooling = nn.MaxPool2d(kernel_size=2)

        self.conv1 = ConvBlock(in_c, nf)
        self.conv2 = ConvBlock(nf, nf*2)
        self.conv3 = ConvBlock(nf*2, nf*4)
        self.conv4 = ConvBlock(nf*4, nf*8)
        self.conv5 = ConvBlock(nf*8, nf*8)

        self.dense = DenseDilatedBlock(nf*8, nf*8)

        self.up1 = UpMerge(nf*8, nf*4)
        self.up2 = UpMerge(nf*4, nf*2)
        self.up3 = UpMerge(nf*2, nf)

        self.out = nn.Sequential(
            nn.Conv2d(nf, out_c, kernel_size=1, stride=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        c1 = self.conv1(x)
        p1 = self.pooling(c1)

        c2 = self.conv2(p1)
        p2 = self.pooling(c2)

        c3 = self.conv3(p2)
        p3 = self.pooling(c3)

        c4 = self.conv4(p3)

        c5 = self.conv5(c4)

        d = self.dense(c5)

        u1 = self.up1(d, c3)
        u2 = self.up2(u1, c2)
        u3 = self.up3(u2, c1)

        out = self.out(u3)

        return out


class Discriminator(nn.Module):
    def __init__(self, args, in_c=4):
        super().__init__()
        self.args = args
        nf = self.args.nf

        self.pooling = nn.MaxPool2d(kernel_size=2)

        self.conv1 = ConvBlock(in_c, nf, 0.1)
        self.conv2 = ConvBlock(nf, nf*2, 0.1)
        self.conv3 = ConvBlock(nf*2, nf*4, 0.2)
        self.conv4 = ConvBlock(nf*4, nf*8, 0.2)
        self.conv5 = ConvBlock(nf*8, nf*16, 0.3)

        self.out = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=1),
            nn.Sigmoid()
        )

    def forward(self, image, mask):
        x = torch.cat((image, mask), 1)

        c1 = self.conv1(x)
        p1 = self.pooling(c1)

        c2 = self.conv2(p1)
        p2 = self.pooling(c2)

        c3 = self.conv3(p2)
        p3 = self.pooling(c3)

        c4 = self.conv4(p3)
        p4 = self.pooling(c4)

        c5 = self.conv5(p4)
        out = self.out(c5)

        return out


class Discriminator1(nn.Module):
    def __init__(self, args, in_c=4):
        super().__init__()
        self.args = args
        nf = self.args.nf

        self.pooling = nn.MaxPool2d(kernel_size=2)

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_c, nf, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(nf, nf*2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(nf*2),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(nf*2, nf*4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(nf*4),
            nn.LeakyReLU(inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(nf*4, nf*8, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(nf*8),
            nn.LeakyReLU(inplace=True)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(nf*8, nf*16, kernel_size=3, stride=1, padding=1, bias=False),
        )

        self.out = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=1),
            nn.Sigmoid()
        )

    def forward(self, image, mask):
        x = torch.cat((image, mask), 1)

        c1 = self.conv1(x)
        p1 = self.pooling(c1)

        c2 = self.conv2(p1)
        p2 = self.pooling(c2)

        c3 = self.conv3(p2)
        p3 = self.pooling(c3)

        c4 = self.conv4(p3)
        p4 = self.pooling(c4)

        c5 = self.conv5(p4)
        out = self.out(c5)

        return out


class Discriminator2(nn.Module):
    def __init__(self, args, in_c=1):
        super().__init__()
        self.args = args
        nf = self.args.nf

        self.pooling = nn.MaxPool2d(kernel_size=2)

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_c, nf, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(nf, nf*2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(nf*2),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(nf*2, nf*4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(nf*4),
            nn.LeakyReLU(inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(nf*4, nf*8, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(nf*8),
            nn.LeakyReLU(inplace=True)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(nf*8, nf*16, kernel_size=3, stride=1, padding=1, bias=False),
        )

        self.out = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        c1 = self.conv1(x)
        p1 = self.pooling(c1)

        c2 = self.conv2(p1)
        p2 = self.pooling(c2)

        c3 = self.conv3(p2)
        p3 = self.pooling(c3)

        c4 = self.conv4(p3)
        p4 = self.pooling(c4)

        c5 = self.conv5(p4)
        out = self.out(c5)

        return out