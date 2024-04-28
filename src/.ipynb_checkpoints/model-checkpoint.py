import torch
import torch.nn as nn
import torch.nn.functional as F
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
        x = torch.cat([x, skip], axis=1)
        x = self.conv(x)

        return x

class VisionTransformer(nn.Module):
    def __init__(self, in_channels, patch_size=16, embed_dim=256, num_heads=8, num_layers=4):
        super(VisionTransformer, self).__init__()

        # Patch embedding layer
        self.patch_embed = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        
        # Initialize class token and positional embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = None  # We will initialize it in the forward method
        
        # Transformer encoder
        transformer_encoder_layer = nn.TransformerEncoderLayer(embed_dim, num_heads, dim_feedforward=2048)
        self.transformer_encoder = nn.TransformerEncoder(transformer_encoder_layer, num_layers)

    def forward(self, x):
        B, C, H, W = x.shape
        
        # Patch embedding
        x = self.patch_embed(x)
        x = x.permute(0, 2, 3, 1).view(B, -1, self.patch_embed.out_channels)
        
        # Calculate the number of patches dynamically
        num_patches = x.shape[1]
        
        # Initialize positional embedding if not already initialized
        if self.pos_embed is None or self.pos_embed.shape[1] != num_patches + 1:
            # Create positional embedding tensor with the correct size
            self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, self.patch_embed.out_channels).to(x.device))
        
        # Concatenate class token and add positional embedding
        cls_token = self.cls_token.expand(B, -1, -1).to(x.device)
        x = torch.cat([cls_token, x], dim=1)
        x += self.pos_embed
        
        # Apply transformer encoder
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
        
        dimension = int(64 / (2 ** n_downsampling))
        self.transformer = PixelwiseViT(
            features=in_features,
            n_heads=4,
            n_blocks=4,
            ffn_features=in_features*4,
            embed_features=in_features,
            activ='gelu',
            norm=None,
            image_shape=(in_features, dimension, dimension),
            rezero=True
        )
        self.bottleneck = ConvBlock(in_features, out_features)
        
        # decoder

        decoder = []

        in_features = out_features
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
            nn.Conv2d(ngf*2, out_channels, kernel_size=7, padding=0),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.input(x)

        encoder_features = []
        for module in self.encoder:
            x, skip = module(x)
            encoder_features.append(skip)

        x = self.transformer(x)
        x = self.bottleneck(x)

        for i, module in enumerate(self.decoder):
            skip = encoder_features[-i-1]
            x = module(x, skip)

        x = self.output(x)

        return x



class PositionWiseFFN(nn.Module):
    def __init__(self, features, ffn_features, activ='gelu'):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(features, ffn_features),
            nn.GELU() if activ == 'gelu' else nn.ReLU(),
            nn.Linear(ffn_features, features)
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):

    def __init__(
        self, features, ffn_features, n_heads, activ = 'gelu', norm = None,
        rezero = True, **kwargs
    ):
        super().__init__(**kwargs)

        self.norm1 = nn.LayerNorm((features,))
        self.atten = nn.MultiheadAttention(features, n_heads)

        self.norm2 = nn.LayerNorm((features,))
        self.ffn   = PositionWiseFFN(features, ffn_features, activ)

        self.rezero = rezero

        if rezero:
            self.re_alpha = nn.Parameter(torch.zeros((1, )))
        else:
            self.re_alpha = 1

    def forward(self, x):
        # x: (L, N, features)

        # Step 1: Multi-Head Self Attention
        y1 = self.norm1(x)
        y1, _atten_weights = self.atten(y1, y1, y1)

        y  = x + self.re_alpha * y1

        # Step 2: PositionWise Feed Forward Network
        y2 = self.norm2(y)
        y2 = self.ffn(y2)

        y  = y + self.re_alpha * y2

        return y

    def extra_repr(self):
        return 're_alpha = %e' % (self.re_alpha, )



class TransformerEncoder(nn.Module):
    def __init__(self, features, ffn_features, n_heads, n_blocks, activ, norm, rezero=True):
        super().__init__()
        self.encoder = nn.Sequential(*[
            TransformerBlock(features, ffn_features, n_heads, activ, norm, rezero)
            for _ in range(n_blocks)
        ])

    def forward(self, x):
        y = x.permute((1, 0, 2))
        y = self.encoder(y)
        result = y.permute((1, 0, 2))
        return result


class FourierEmbedding(nn.Module):
    def __init__(self, features, height, width):
        super().__init__()
        self.projector = nn.Linear(2, features)
        self._height = height
        self._width = width

    def forward(self, y, x):
        x_norm = 2 * x / (self._width - 1) - 1
        y_norm = 2 * y / (self._height - 1) - 1
        z = torch.cat((x_norm.unsqueeze(2), y_norm.unsqueeze(2)), dim=2)
        return torch.sin(self.projector(z))


class ViTInput(nn.Module):
    def __init__(self, input_features, embed_features, features, height, width):
        super().__init__()
        x = torch.arange(width).to(torch.float32)
        y = torch.arange(height).to(torch.float32)
        x, y = torch.meshgrid(x, y)
        self.x = x.reshape((1, -1))
        self.y = y.reshape((1, -1))
        self.register_buffer('x_const', self.x)
        self.register_buffer('y_const', self.y)
        self.embed = FourierEmbedding(embed_features, height, width)
        self.output = nn.Linear(embed_features + input_features, features)

    def forward(self, x):
        embed = self.embed(self.y_const, self.x_const)
        embed = embed.expand((x.shape[0], *embed.shape[1:]))
        result = torch.cat([embed, x], dim=2)
        return self.output(result)

class PixelwiseViT(nn.Module):
    def __init__(self, features, n_heads, n_blocks, ffn_features, embed_features, image_shape, activ='gelu',
                 norm='layer', rezero=True):
        super().__init__()
        self.image_shape = image_shape
        self.trans_input = ViTInput(image_shape[0], embed_features, features, image_shape[1], image_shape[2])
        self.encoder = TransformerEncoder(features, ffn_features, n_heads, n_blocks, activ, norm, rezero)
        self.trans_output = nn.Linear(features, image_shape[0])

    def forward(self, x):
        itokens = x.view(*x.shape[:2], -1)
        itokens = itokens.permute((0, 2, 1))
        y = self.trans_input(itokens)
        y = self.encoder(y)
        otokens = self.trans_output(y)
        otokens = otokens.permute((0, 2, 1))
        return otokens.view(*otokens.shape[:2], *self.image_shape[1:])

class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        # Encoder
        self.e11 = nn.Conv2d(3, 48, kernel_size=3, padding=1)
        self.e12 = nn.Conv2d(48, 48, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.e21 = nn.Conv2d(48, 96, kernel_size=3, padding=1)
        self.e22 = nn.Conv2d(96, 96, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.e31 = nn.Conv2d(96, 192, kernel_size=3, padding=1)
        self.e32 = nn.Conv2d(192, 192, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.e41 = nn.Conv2d(192, 384, kernel_size=3, padding=1)
        self.e42 = nn.Conv2d(384, 384, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # PixelwiseViT
        self.pixelwise_vit = PixelwiseViT(
            features=384,
            n_heads=6,
            n_blocks=6,
            ffn_features=1536,
            embed_features=384,
            activ='gelu',
            norm=None,
            image_shape=(384, 4, 4),
            rezero=True  # please replace '_' with the appropriate sizes
        )

        # Decoder
        self.upconv1 = nn.ConvTranspose2d(384, 192, kernel_size=2, stride=2)
        self.d11 = nn.Conv2d(384, 192, kernel_size=3, padding=1)
        self.d12 = nn.Conv2d(192, 192, kernel_size=3, padding=1)

        self.upconv2 = nn.ConvTranspose2d(192, 96, kernel_size=2, stride=2)
        self.d21 = nn.Conv2d(192, 96, kernel_size=3, padding=1)
        self.d22 = nn.Conv2d(96, 96, kernel_size=3, padding=1)

        self.upconv3 = nn.ConvTranspose2d(96, 48, kernel_size=2, stride=2)
        self.d31 = nn.Conv2d(96, 48, kernel_size=3, padding=1)
        self.d32 = nn.Conv2d(48, 48, kernel_size=3, padding=1)

        self.upconv4 = nn.ConvTranspose2d(48, 48, kernel_size=2, stride=2)
        self.d41 = nn.Conv2d(48, 48, kernel_size=3, padding=1)
        self.d42 = nn.Conv2d(48, 48, kernel_size=3, padding=1)

        # Output layer
        self.outconv = nn.Identity()


    def forward(self, x):
        # Encoder
        x1 = self.pool1(F.relu(self.e12(F.relu(self.e11(x)))))
        x2 = self.pool2(F.relu(self.e22(F.relu(self.e21(x1)))))
        x3 = self.pool3(F.relu(self.e32(F.relu(self.e31(x2)))))
        x4 = self.pool4(F.relu(self.e42(F.relu(self.e41(x3)))))
        print("x4",x4.size())
        # PixelwiseViT
        x5 = self.pixelwise_vit(x4)
        print("x5", x5.size())

        # Decoder
        x = torch.cat([self.upconv1(x5), F.interpolate(x4, scale_factor=2)], 1)
        print("Post first concat, pre d11/d12: ", x.shape)
        x = F.relu(self.d11(x))
        x = F.relu(self.d12(x))
        print("Post d11/d12: ", x.shape)
        
        x = torch.cat([self.upconv2(x), F.interpolate(x3, scale_factor=2)], 1)
        print("Post second concat, pre d21/d22: ", x.shape)
        x = F.relu(self.d21(x))
        x = F.relu(self.d22(x))
        print("Post d21/d22: ", x.shape)
        
        x = torch.cat([self.upconv3(x), F.interpolate(x2, scale_factor=2)], 1)
        print("Post third concat, pre d31/d32: ", x.shape)
        x = F.relu(self.d31(x))
        x = F.relu(self.d32(x))
        print("Post d31/d32: ", x.shape)
        
        x = torch.cat([self.upconv4(x), F.interpolate(x1, scale_factor=2)], 1)
        print("Post fourth concat, pre d41/d42: ", x.shape)
        x = F.relu(self.d41(x))
        x = F.relu(self.d42(x))
        print("Post d41/d42: ", x.shape)
        
        # Output layer
        x = self.outconv(x)
        print("Final output: ", x.shape)
        
        return x



class ViTUNetGenerator(nn.Module):
    def __init__(
        self, image_shape, unet_features_list, activ_name='leaky_relu',
        features=384, n_heads=6, n_blocks=4, ffn_features=128, embed_features=384,
        unet_downsample='conv', unet_upsample='upsample-conv', unet_rezero=False,
        rezero=True, activ_output=None, **kwargs
    ):
        super().__init__(**kwargs)

        # UNet with given image shape and feature list
        self.unet = UNet()
        
        # Output activation
        self.output_activation = nn.Identity()

    def forward(self, x):
        x = self.unet(x)
        x = self.output_activation(x)
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
