import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, in_features):
        super().__init__()

        self.model = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
            nn.ReLU(inplace=True)
       )

    def forward(self, x):
        return x + self.model(x)
    
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.model(x)

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = ConvBlock(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(x)
        skip = x

        x = self.pool(x)

        return x, skip

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2, padding=0),
            nn.ReLU()
        )
        self.conv = ConvBlock(in_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)

        if x.shape[2:] != skip.shape[2:]:
            skip = F.interpolate(skip, size=x.shape[2:], mode='nearest')

        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)

        return x

class VisionTransformer(nn.Module):
    def __init__(self, in_channels, patch_size=16, embed_dim=256, num_heads=8, num_layers=4):
        super(VisionTransformer, self).__init__()

        self.patch_embed = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, (64 // patch_size) ** 2 + 1, embed_dim))
        self.transformer_encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(embed_dim, num_heads, dim_feedforward=2048), num_layers)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.patch_embed(x)
        x = x.permute(0, 2, 3, 1).view(B, -1, C)
        x = torch.cat([self.cls_token.expand(B, -1, -1), x], dim=1)
        x += self.pos_embed
        x = self.transformer_encoder(x)
        return x

class CycleGANGenerator(nn.Module):
    def __init__(self, in_channels, out_channels, ngf=64, n_downsampling=2, n_layers=6):
        super().__init__()

        # input
        
        self.input = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, ngf, kernel_size=7, padding=0),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(inplace=True) 
        )

        # encoder

        encoder = []

        in_features = ngf
        out_features = ngf * 2
        
        for _ in range(n_downsampling):
            encoder += [
                nn.Conv2d(in_features, out_features, kernel_size=3, stride=2, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True)
            ]
            
            in_features = out_features
            out_features = in_features*2

        self.encoder = nn.Sequential(*encoder)
        
        # transformer

        transformer = []

        for _ in range(n_layers):
            transformer += [
                ResBlock(in_features)
            ]

        self.transformer = nn.Sequential(*transformer)
        
        # decoder

        decoder = []

        out_features = in_features//2
        
        for _ in range(n_downsampling):
            decoder += [
                nn.ConvTranspose2d(in_features, out_features, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True)
            ]
            
            in_features = out_features
            out_features = in_features//2

        self.decoder = nn.Sequential(*decoder)

        # output

        self.output = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, out_channels, kernel_size=7, padding=0),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.input(x)
        x = self.encoder(x)
        x = self.transformer(x)
        x = self.decoder(x)
        x = self.output(x)

        return x

class UVCGANGenerator(nn.Module):
    def __init__(self, in_channels, out_channels, ngf=64, n_downsampling=2, n_layers=6):
        super().__init__()

        # input

        self.input = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, ngf, kernel_size=7, padding=0),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(inplace=True)
        )

        # encoder

        encoder = []
        
        in_features = ngf
        out_features = ngf * 2
        
        for _ in range(n_downsampling):
            encoder += [
                EncoderBlock(in_features, out_features)
            ]
            
            in_features = out_features
            out_features = in_features*2

        self.encoder = nn.Sequential(*encoder)
        
        # transformer
        
        self.transformer = VisionTransformer(in_channels=in_features)
        
        # decoder

        decoder = []

        out_features = in_features//2
        
        for _ in range(n_downsampling):
            decoder += [
                DecoderBlock(in_features, out_features)
            ]
            
            in_features = out_features
            out_features = in_features//2

        self.decoder = nn.Sequential(*decoder)

        # output

        self.output = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, out_channels, kernel_size=7, padding=0),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.input(x)

        encoder_features = []
        for module in self.encoder:
            x, skip = module(x)
            encoder_features.append(skip)

        x = self.transformer(x)

        for i, module in enumerate(self.decoder):
            skip = encoder_features[-i-1]
            x = module(x, skip)

        x = self.output(x)

        return x

class Discriminator(nn.Module):
    def __init__(self, in_channels, ndf=64, n_layers=3):
        super().__init__()

        model = [
            nn.Conv2d(in_channels, ndf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True)
        ]

        in_features = ndf
        out_features = ndf * 2
        
        for _ in range(1, n_layers):
            model += [
                nn.Conv2d(in_features, out_features, kernel_size=4, stride=2, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.LeakyReLU(0.2, True)
            ]

            in_features = out_features
            out_features = in_features * 2
        
        model += [
            nn.Conv2d(in_features, out_features, kernel_size=4, stride=1, padding=1),
            nn.InstanceNorm2d(out_features),
            nn.LeakyReLU(0.2, True)
        ]

        model += [
            nn.Conv2d(out_features, 1, kernel_size=4, stride=1, padding=1)
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        x = self.model(x)
        return F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1)
