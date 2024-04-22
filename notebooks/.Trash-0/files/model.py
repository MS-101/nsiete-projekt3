import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features)
       )

    def forward(self, x):
        return x + self.conv_block(x)    

class Generator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_downsampling=2, n_layers=6):
        super(Generator, self).__init__()

        encoder = []
        
        # input layer
        
        encoder += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(inplace=True) 
        ]

        # downsampling
        
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

        transformer = []

        # residual blocks
        
        for _ in range(n_layers):
            transformer += [
                ResidualBlock(in_features)
            ]

        decoder = []

        # upsampling

        out_features = in_features//2
        
        for _ in range(n_downsampling):
            decoder += [
                nn.ConvTranspose2d(in_features, out_features, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True)
            ]
            
            in_features = out_features
            out_features = in_features//2

        # output layer

        decoder += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
            nn.Tanh()
        ]

        self.encoder = nn.Sequential(*encoder)
        self.transformer = nn.Sequential(*transformer)
        self.decoder = nn.Sequential(*decoder)

    def forward(self, x):
        x = self.encoder(x)
        x = self.transformer(x)
        x = self.decoder(x)
        return x

class Discriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3):
        super(Discriminator, self).__init__()

        model = [
            nn.Conv2d(input_nc, ndf, kernel_size=4, stride=2, padding=1),
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
            nn.Conv2d(out_features, 1, kernel_size=4, padding=1)
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)