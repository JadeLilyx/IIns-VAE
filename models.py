import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


def weights_init_normal(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d):  # classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d):  # classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


class LambdaLR:
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert (n_epochs - decay_start_epoch) > 0, "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch) / (self.n_epochs - self.decay_start_epoch)


# ===================================
#         Basic Architecture
# ===================================

class Encoder(nn.Module):
    def __init__(self, conv_type=1, dim=4, n_residual=3, n_downsample=4, style_dim=8, out_dim=2, expand=False):
        super(Encoder, self).__init__()
        self.conv_type = conv_type
        self.expand = expand
        self.latent_dim = style_dim
        if self.conv_type == 1:
            self.range_encoder = RangeEncoder1d(dim, n_residual, n_downsample, out_dim)  # dim=4, n_downsample=4
            self.env_encoder = EnvEncoder1d(dim * 4, n_downsample - 2, style_dim)  # dim=16, n_downsample=2
        elif self.conv_type == 2:
            self.range_encoder = RangeEncoder2d(dim, n_residual, n_downsample, out_dim)
            self.env_encoder = EnvEncoder2d(dim * 4, n_downsample - 2, style_dim)
        else:
            # not available yet
            self.range_encoder = RangeEncoder2dNoExpand(dim, n_residual, n_downsample, out_dim)
            self.env_encoder = EnvEncoder2dNoExpand(dim * 4, n_downsample - 2, style_dim)

    def forward(self, x):
        # conv on 152 or (152, 1), (152, 152)
        if self.conv_type == 1:
            x = x.view(x.size(0), 1, x.size(1))  # (B, 1, 152)
            # x = x.unsqueeze(1)
        else:
            x = x.view(x.size(0), 1, x.size(1), 1).expand((x.size(0), 1, x.size(1), x.size(1))) if self.expand \
                else x.view(x.size(0), 1, x.size(1), 1)
            # (B, 1, 152, 152) or (B, 1, 152, 1) as in ewine

        range_code = self.range_encoder(x)  # (B, 1, 152, 152/1/none) -> (B, 2, 8, 8/1/none)
        env_code, env_code_rv, kl_div = self.env_encoder(x)  # (B, 1, 152, 152/1/none) -> (B, 8, 1, 1/none)
        return range_code, env_code, env_code_rv, kl_div  # env_code_rv only for cls, or try env_code

    def sample(self, n):
        return torch.randn(n, self.latent_dim)


class Decoder(nn.Module):
    def __init__(self, conv_type=1, dim=4, n_residual=3, n_upsample=4, style_dim=8, in_dim=152, out_dim=2, expand=False):
        super(Decoder, self).__init__()

        self.conv_type = conv_type
        self.expand = expand
        if conv_type == 1:
            self.decoder = Decoder1d(dim, n_residual, n_upsample, in_dim, out_dim, style_dim)
        elif conv_type == 2:
            self.decoder = Decoder2d(dim, n_residual, n_upsample, in_dim, out_dim, style_dim)
        else:
            # not available yet
            self.decoder = Decoder2dNoExpand(dim, n_residual, n_upsample, in_dim, out_dim, style_dim)

    def forward(self, range_code, env_code):
        x_recon = self.decoder(range_code, env_code)  # (2, 8) + (8, 1) -> (1, 152)

        # restore original size (B, 152)
        # if self.conv_dim == 1:
        #     x_recon = x_recon.view(x_recon.size(0), -1)
        #     # x_recon = x_recon.squeeze()
        # else:
        #     x_recon = x_recon[:, :, :, 0].squeeze() if self.expand else x_recon.squeeze()
        x_recon = x_recon[:, :, :, 0].squeeze() if self.expand else x_recon.squeeze()  # (B, 152)
        return x_recon


class Restorer(nn.Module):
    """Constrain range code with ranging error."""
    # can later provide the option for dist_m, (B, 2, 8, (8/1)) -> (B, 1/2)
    def __init__(self, code_shape, soft=False, filters=64, conv_type=1, expand=False, net_type='Linear'):
        super(Restorer, self).__init__()

        self.soft = soft
        if net_type == 'Linear':
            self.restorer = RestorerLinear(code_shape, soft)
        elif net_type == 'Conv1d':
            self.restorer = RestorerConv1d(code_shape, soft, filters, conv_type, expand)
        elif net_type == 'Conv2d':
            self.restorer = RestorerConv2d(code_shape, soft, filters, conv_type, expand)
        else:
            raise ValueError("Unknown network type for Restorer.")

    def forward(self, range_code):
        err_est = self.restorer(range_code)
        return err_est


class Classifier(nn.Module):
    """Constrain range code with ranging error."""
    # can later provide the option for dist_m, (B, 2, 8, (8/1)) -> (B, 1/2)
    def __init__(self, env_dim, num_classes, filters=16, net_type='Linear'):
        super(Classifier, self).__init__()

        if net_type == 'Linear':
            self.classifier = ClassifierLinear(env_dim, num_classes, filters)
        elif net_type == 'Conv1d':
            self.classifier = ClassifierConv1d(env_dim, num_classes, filters)
        elif net_type == 'Conv2d':
            self.classifier = ClassifierConv2d(env_dim, num_classes, filters)
        else:
            raise ValueError("Unknown network type for Classifier.")

    def forward(self, env_code):
        label_est = self.classifier(env_code)
        return label_est


# =================================================
#           Encoders (x -> codes)
# =================================================


class RangeEncoder1d(nn.Module):
    def __init__(self, dim=4, n_residual=3, n_downsample=4, out_dim=2):
        super(RangeEncoder1d, self).__init__()

        # Initialize input to 128 dim, (1, 152) -> (1, 128)
        layers = []
        layers += [nn.AdaptiveAvgPool1d(128)]

        # Initial convolution block, (1, 128) -> (4, 128)
        layers += [
            nn.ReflectionPad1d(3),  # (1, 134), 128+6
            nn.Conv1d(1, dim, 7),  # (4, 128), 134-7+1
            nn.InstanceNorm1d(dim),
            nn.ReLU(inplace=True),
        ]

        # Downsampling
        for _ in range(n_downsample):  # (4, 128) -> (8, 64) -> (16, 32) -> (32, 16) -> (64, 8)
            layers += [
                nn.Conv1d(dim, dim * 2, 4, stride=2, padding=1),  # (128-4+2)/2+1
                nn.InstanceNorm1d(dim * 2),
                nn.ReLU(inplace=True),
            ]
            dim *= 2

        # Residual blocks
        for _ in range(n_residual):
            layers += [ResidualBlock1d(dim, norm="in")]

        # Output layer, (64, 8) -> (2, 8)
        # dim = dim * 2 ** n_downsample  # 64
        layers += [nn.Conv1d(dim, out_dim, 1, 1, 0), nn.ReLU(inplace=True)]

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)  # (1, 152) -> (2, 8)


class RangeEncoder2d(nn.Module):
    def __init__(self, dim=4, n_residual=4, n_downsample=4, out_dim=2):
        super(RangeEncoder2d, self).__init__()

        # Initialize input dim to 128, (1, 152, 152) -> (1, 128, 128)
        layers = []
        layers += [nn.AdaptiveAvgPool2d(128)] 

        # Initial convolution block
        layers += [  # (1, 128, 128) -> (4, 128, 128)
            nn.ReflectionPad2d(3),
            nn.Conv2d(1, dim, 7),
            nn.InstanceNorm2d(dim),
            nn.ReLU(inplace=True),
        ]

        # Downsampling
        for _ in range(n_downsample):  # (4, 128, 128) -> (8, 64, 64) -> (16, 32, 32) -> (32, 16, 16) -> (64, 8, 8)
            layers += [
                nn.Conv2d(dim, dim * 2, 4, stride=2, padding=1),
                nn.InstanceNorm2d(dim * 2),
                nn.ReLU(inplace=True),
            ]
            dim *= 2

        # Residual blocks
        for _ in range(n_residual):  # (64, 8, 8)
            layers += [ResidualBlock2d(dim, norm="in")]

        # Output layer, (64, 8, 8) -> (2, 8, 8)
        # dim = dim * 2 ** n_downsample  # already there
        layers += [nn.Conv2d(dim, out_dim, 1, 1, 0), nn.ReLU(inplace=True)]

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)  # (b, 2, 8, 8)


class RangeEncoder2dNoExpand(nn.Module):
    """Not available yet temporally."""
    def __init__(self, dim=4, n_residual=4, n_downsample=4, out_dim=2):
        super(RangeEncoder2dNoExpand, self).__init__()

        # Initialize input dim 128, (1, 152, 1) -> (1, 128, 1)
        layers = []
        layers += [nn.AdaptiveAvgPool2d((128, 1))]

        # Initial convolution block
        layers += [  # (1, 128, 1) -> (4, 128, 1)
            # nn.ReflectionPad2d(3),
            # nn.Conv2d(1, dim, kernel_size=7),
            nn.Conv2d(1, dim, 1, 1, 0),
            # nn.InstanceNorm2d(dim),
            nn.ReLU(inplace=True),
        ]

        # Downsampling
        for _ in range(n_downsample):  # (4, 128, 1) -> (8, 64, 1) -> (16, 32, 1) -> (32, 16, 1) -> (64, 8, 1)
            layers += [
                nn.Conv2d(dim, dim * 2, (4, 1), stride=(2, 1), padding=(1, 0)),
                nn.InstanceNorm2d(dim * 2),
                nn.ReLU(inplace=True),
            ]
            dim *= 2

        # Residual blocks
        for _ in range(n_residual):  # (64, 8, 1)
            layers += [ResidualBlock2dNoExpand(dim, norm="in")]

        # Output layer, (64, 8, 1) -> (2, 8, 1)
        layers += [nn.Conv2d(dim, out_dim, 1, 1, 0), nn.ReLU(inplace=True)]

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)  # (b, 2, 8, 1)


class EnvEncoder1d(nn.Module):
    def __init__(self, dim=16, n_downsample=2, style_dim=8):
        super(EnvEncoder1d, self).__init__()

        # Initialize input dim, (1, 152) -> (1, 128)
        layers = []
        layers += [nn.AdaptiveAvgPool1d(128)]

        # Initial conv block, (1, 128) -> (16, 128)
        layers += [nn.ReflectionPad1d(3), nn.Conv1d(1, dim, 7), nn.ReLU(inplace=True)]

        # Downsampling
        for _ in range(2):  # (16, 128) -> (32, 64) -> (64, 32)
            layers += [nn.Conv1d(dim, dim * 2, 4, stride=2, padding=1), nn.ReLU(inplace=True)]
            dim *= 2

        # Downsampling with constant depth
        for _ in range(n_downsample - 2):  # (64, 32) -> (64, 16)
            layers += [nn.Conv1d(dim, dim, 4, stride=2, padding=1), nn.ReLU(inplace=True)]

        # Average pool and output layer, (64, 16) -> (64, 1) -> (8, 1)
        layers += [nn.AdaptiveAvgPool1d(1), nn.Conv1d(dim, style_dim, 1, 1, 0)]

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        # print("env code shape: ", self.model(x).shape)  # (b, 8)/(b, style_dim)
        cat = self.model(x)  # (8, 1)
        latent_dim = cat.shape[1] // 2  # 4
        mu, log_sigma = torch.split(cat, latent_dim, dim=1)
        noise = torch.randn_like(mu)
        latent_code = noise * log_sigma.exp() + mu  # (4, 1)
        kl_div = self._loss(mu, log_sigma)  # calculate with 8dim instead of 4dim randn

        return cat, latent_code, kl_div

    def _loss(self, mu, log_sigma):
        kl_div = 0.5 * torch.sum((2 * log_sigma).exp() + mu ** 2 - 1 - 2 * log_sigma, dim=1)
        kl_div = kl_div.mean()  # account for batch size

        return kl_div

    # def sample(self, n):
    #     return torch.randn(n, self.latent_dim)


class EnvEncoder2d(nn.Module):
    def __init__(self, dim=16, n_downsample=2, style_dim=8):
        super(EnvEncoder2d, self).__init__()
        
        # Initialize input dim, (1, 152， 152) -> (1, 128， 128)
        layers = []
        layers += [nn.AdaptiveAvgPool2d(128)]

        self.latent_dim = style_dim // 2

        # Initial conv block, (1, 128, 128) -> (16, 128, 128)
        layers += [nn.ReflectionPad2d(3), nn.Conv2d(1, dim, 7), nn.ReLU(inplace=True)]

        # Downsampling, (16, 128, 128) -> (32, 64, 64) -> (64, 32, 32)
        for _ in range(2):
            layers += [nn.Conv2d(dim, dim * 2, 4, stride=2, padding=1), nn.ReLU(inplace=True)]
            dim *= 2

        # Downsampling with constant depth
        for _ in range(n_downsample - 2):  # (64, 32, 32) -> (64, 16, 16) / 31
            layers += [nn.Conv2d(dim, dim, 4, stride=2, padding=1), nn.ReLU(inplace=True)]

        # Average pool and output layer, (64, 16, 16) -> (64, 1, 1) -> (8, 1, 1)
        layers += [nn.AdaptiveAvgPool2d(1), nn.Conv2d(dim, style_dim, 1, 1, 0)]

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        # print("env code shape: ", self.model(x).shape)  # (b, 8)/(b, style_dim)
        cat = self.model(x)  # (8, 1, 1)
        latent_dim = cat.shape[1] // 2
        mu, log_sigma = torch.split(cat, latent_dim, dim=1)  # (4, 1, 1)
        noise = torch.randn_like(mu)
        latent_code = noise * log_sigma.exp() + mu  # (4, 1, 1)
        kl_div = self._loss(mu, log_sigma)

        return cat, latent_code, kl_div

    def _loss(self, mu, log_sigma):
        kl_div = 0.5 * torch.sum((2 * log_sigma).exp() + mu ** 2 - 1 - 2 * log_sigma, dim=1)
        kl_div = kl_div.mean()  # account for batch size

        return kl_div

    # def sample(self, n):
    #     return torch.randn(n, self.latent_dim)


class EnvEncoder2dNoExpand(nn.Module):
    """Not available yet temporally."""    
    def __init__(self, dim=16, n_downsample=2, style_dim=8):
        super(EnvEncoder2dNoExpand, self).__init__()

        self.latent_dim = style_dim // 2

        # Initialize input, (1, 152, 1) -> (1, 128, 1)
        layers = [nn.AdaptiveAvgPool2d((128, 1))]

        # Initial conv block, (1, 128, 1) -> (16, 128, 1)
        layers += [nn.ReflectionPad2d(3), nn.Conv2d(1, dim, 7), nn.ReLU(inplace=True)]

        # Downsampling, (16, 128, 1) -> (32, 64, 1) -> (64, 32, 1)
        for _ in range(2):
            layers += [nn.Conv2d(dim, dim * 2, (4, 1), stride=(2, 1), padding=(1, 0)), nn.ReLU(inplace=True)]
            dim *= 2

        # Downsampling with constant depth
        for _ in range(n_downsample - 2):  # (64, 32, 1) -> (64, 16, 1)
            layers += [nn.Conv2d(dim, dim, (4, 1), stride=(2, 1), padding=(1, 0)), nn.ReLU(inplace=True)]

        # Average pool and output layer, (64, 16, 1) -> (64, 1, 1) -> (8, 1, 1)
        layers += [nn.AdaptiveAvgPool2d(1), nn.Conv2d(dim, style_dim, 1, 1, 0)]

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        # print("env code shape: ", self.model(x).shape)  # (b, 8)/(b, style_dim)
        cat = self.model(x)  # (8, 1, 1)
        latent_dim = cat.shape[1] // 2
        mu, log_sigma = torch.split(cat, latent_dim, dim=1)  # (4, 1, 1)
        noise = torch.randn_like(mu)
        latent_code = noise * log_sigma.exp() + mu  # (4, 1, 1)
        kl_div = self._loss(mu, log_sigma)

        return cat, latent_code, kl_div

    def _loss(self, mu, log_sigma):
        kl_div = 0.5 * torch.sum((2 * log_sigma).exp() + mu ** 2 - 1 - 2 * log_sigma, dim=1)
        kl_div = kl_div.mean()  # account for batch size

        return kl_div

    # def sample(self, n):
    #     return torch.randn(n, self.latent_dim)


# ==================================================
#          Decoders (codes -> x_recon)
# ==================================================


class Decoder1d(nn.Module):
    def __init__(self, dim=4, n_residual=3, n_upsample=4, in_dim=152, out_dim=2, style_dim=8):
        super(Decoder1d, self).__init__()

        layers = []
        dim = dim * 2 ** n_upsample  # 64

        # Initialize input, (2, 8) -> (64, 8)
        layers += [nn.Conv1d(out_dim, dim, 1, 1, 0), nn.ReLU(inplace=True)]
        
        # Residual blocks
        for _ in range(n_residual):  # (64, 8)
            layers += [ResidualBlock1d(dim, norm="adain")]

        # Upsampling
        for _ in range(n_upsample):  # (64, 8) -> (32, 16) -> (16, 32) -> (8, 64) -> (4, 128)
            layers += [
                nn.Upsample(scale_factor=2),  # (64, 16)
                nn.Conv1d(dim, dim // 2, 5, stride=1, padding=2),  # (32, 16)
                LayerNorm(dim // 2),
                nn.ReLU(inplace=True),
            ]
            dim = dim // 2

        # Output layer, (4, 128) -> (1, 128)
        layers += [
            nn.ReflectionPad1d(3),
            nn.Conv1d(dim, 1, 7),
            nn.Tanh()
        ]

        layers += [nn.AdaptiveAvgPool1d(in_dim)]  # (1, 152)

        self.model = nn.Sequential(*layers)

        # Initiate mlp (predict AdaIN parameters)
        num_adain_params = self.get_num_adain_params()
        self.mlp = MLP(style_dim, num_adain_params)

    def get_num_adain_params(self):
        """Return the number of AdaIN params needed by the model"""
        num_adain_params = 0
        for m in self.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm1d":
                num_adain_params += 2 * m.num_features
        return num_adain_params

    def assign_adain_params(self, adain_params):
        """Assign the adain_params to the AdaIN layers in model"""
        for m in self.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm1d":
                # Extract mean and std predictions
                mean = adain_params[:, : m.num_features]
                std = adain_params[:, m.num_features : 2 * m.num_features]
                # Update bias and weight
                m.bias = mean.contiguous().view(-1)
                m.weight = std.contiguous().view(-1)
                # Move pointer
                if adain_params.size(1) > 2 * m.num_features:
                    adain_params = adain_params[:, 2 * m.num_features :]

    def forward(self, range_code, env_code):
        # Update AdaIN parameters by MLP prediction based off env_code
        self.assign_adain_params(self.mlp(env_code))  # (4, 1) -> (num_adain_params, 1)
        # combine range_code in for the final output
        x_recon = self.model(range_code)  # (2, 8) -> (1, 152)
        return x_recon


class Decoder2d(nn.Module):
    def __init__(self, dim=4, n_residual=3, n_upsample=4, in_dim=152, out_dim=2, style_dim=8):
        super(Decoder2d, self).__init__()

        layers = []
        dim = dim * 2 ** n_upsample

        # Initialize input, (2, 8, 8) -> (64, 8, 8)
        layers += [nn.Conv2d(out_dim, dim, 1, 1, 0), nn.ReLU(inplace=True)]

        # Residual blocks, (64, 8, 8)
        for _ in range(n_residual):
            layers += [ResidualBlock2d(dim, norm="adain")]

        # Upsampling, (64, 8, 8) -> (32, 16, 16) -> (16, 32, 32) -> (8, 64, 64) -> (4, 128, 128)
        for _ in range(n_upsample):
            layers += [
                nn.Upsample(scale_factor=2),
                nn.Conv2d(dim, dim // 2, 5, stride=1, padding=2),
                LayerNorm(dim // 2),
                nn.ReLU(inplace=True),
            ]
            dim = dim // 2

        # Output layer
        layers += [  # (4, 128, 128) -> (1, 128, 128)
            nn.ReflectionPad2d(3),
            nn.Conv2d(dim, 1, 7),
            nn.Tanh()
        ]

        layers += [nn.AdaptiveAvgPool2d(in_dim)]  # (1, 152, 152)

        self.model = nn.Sequential(*layers)

        # Initialize mlp (predict AdaIN parameters)
        num_adain_params = self.get_num_adain_params()
        self.mlp = MLP(style_dim, num_adain_params)  # (4, 1, 1) -> (num, 1, 1)

    def get_num_adain_params(self):
        """Return the number of AdaIN parameters needed by the model"""
        num_adain_params = 0
        for m in self.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
                num_adain_params += 2 * m.num_features
        return num_adain_params

    def assign_adain_params(self, adain_params):
        """Assign the adain_params to the AdaIN layers in model"""
        for m in self.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
                # Extract mean and std predictions
                mean = adain_params[:, : m.num_features]
                std = adain_params[:, m.num_features : 2 * m.num_features]
                # Update bias and weight
                m.bias = mean.contiguous().view(-1)
                m.weight = std.contiguous().view(-1)
                # Move pointer
                if adain_params.size(1) > 2 * m.num_features:
                    adain_params = adain_params[:, 2 * m.num_features :]

    def forward(self, range_code, env_code):
        # Update AdaIN parameters by MLP prediction based off style code
        self.assign_adain_params(self.mlp(env_code))
        x_recon = self.model(range_code)
        return x_recon


class Decoder2dNoExpand(nn.Module):
    """Not available yet temporally."""
    def __init__(self, dim=64, n_residual=3, n_upsample=2, in_dim=152, out_dim=2, style_dim=8):
        super(Decoder2dNoExpand, self).__init__()

        layers = []
        dim = dim * 2 ** n_upsample

        # Initialize input, (2, 8, 1) -> (64, 8, 1)
        layers += [nn.Conv2d(out_dim, dim, 1, 1, 0), nn.ReLU(inplace=True)]

        # Residual blocks, (64, 8, 1)
        for _ in range(n_residual):
            layers += [ResidualBlock2d(dim, norm="adain")]

        # Upsampling, (64, 8, 1) -> (32, 16, 1) -> (16, 32, 1) -> (8, 64, 1) -> (4, 128, 1)
        for _ in range(n_upsample):
            layers += [
                nn.Upsample(scale_factor=(2, 1)),
                nn.Conv2d(dim, dim // 2, (5, 1), stride=1, padding=(2, 0)),
                LayerNorm(dim // 2),
                nn.ReLU(inplace=True),
            ]
            dim = dim // 2

        # Output layer
        layers += [  # (4, 128, 1) -> (1, 128, 1)
            nn.ReflectionPad2d((3, 1)),
            nn.Conv2d(dim, 1, (7, 1), stride=1, padding=0),
            nn.Tanh()
        ]

        layers += [nn.AdaptiveAvgPool2d((in_dim, 1))]  # (1, 152, 1)

        self.model = nn.Sequential(*layers)

        # Initialize mlp (predict AdaIN parameters)
        num_adain_params = self.get_num_adain_params()
        self.mlp = MLP(style_dim, num_adain_params)  # (4, 1, 1) -> (num, 1, 1)

    def get_num_adain_params(self):
        """Return the number of AdaIN parameters needed by the model"""
        num_adain_params = 0
        for m in self.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
                num_adain_params += 2 * m.num_features
        return num_adain_params

    def assign_adain_params(self, adain_params):
        """Assign the adain_params to the AdaIN layers in model"""
        for m in self.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
                # Extract mean and std predictions
                mean = adain_params[:, : m.num_features]
                std = adain_params[:, m.num_features : 2 * m.num_features]
                # Update bias and weight
                m.bias = mean.contiguous().view(-1)
                m.weight = std.contiguous().view(-1)
                # Move pointer
                if adain_params.size(1) > 2 * m.num_features:
                    adain_params = adain_params[:, 2 * m.num_features :]

    def forward(self, range_code, env_code):
        # Update AdaIN parameters by MLP prediction based off style code
        self.assign_adain_params(self.mlp(env_code))
        x_recon = self.model(range_code)
        return x_recon

# ==================================================
#         Restorers (range_code -> err_est)
# ==================================================


class RestorerLinear(nn.Module):
    def __init__(self, code_shape, soft=False):
        super(RestorerLinear, self).__init__()
        self.soft = soft
        self.layers = nn.Sequential(
            # Flatten
            nn.Linear(int(np.prod(code_shape)), 512),  # +1 for dist_gt, 2*8/2*8*8 -> 512
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 256),
            # nn.BatchNorm1d(256, 0.8),  # require batch_size>1
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Linear(256, 2),
        )

        self.linear_layer1 = nn.Linear(256, 1)
        self.linear_layer2 = nn.Linear(256, 2)  # nn.Softmax())
        # self.sigmoid = nn.Sigmoid()  # not 0~1

    def reparameterization(self, mu, logvar):
        Tensor = torch.cuda.FloatTensor if mu.is_cuda else torch.FloatTensor
        std = torch.exp(logvar / 2)
        sampled_z = Variable(Tensor(np.random.normal(0, 1, (mu.size(0), 1))))
        z = sampled_z * std + mu
        return z

    def forward(self, range_code):
        code_flat = range_code.view(range_code.size(0), -1)
        # 1d: (b, 2, 8) -> (b, 16)
        # 2d: (b, 2, 8, 8) -> (b, 128)
        err_est = self.layers(code_flat)
        # 1d: (b, 16) -> (b, 512) -> (b, 256)
        # 2d: (b, 128) -> (b, 512) -> (b, 256)

        if self.soft:
            err_est = self.linear_layer2(err_est)  # 256 -> 2
            mu = err_est[:, 0]
            logvar = err_est[:, 1]
            err_est = self.reparameterization(mu, logvar)  # (b, 1)
        else:
            err_est = self.linear_layer1(err_est)  # 256 -> 1

        return err_est


class RestorerConv1d(nn.Module):
    def __init__(self, code_shape, soft=False, filters=64, conv_type=1, expand=False):
        super(RestorerConv1d, self).__init__()
        self.soft = soft
        self.conv_type = conv_type
        self.expand = expand

        def conv_block(in_filters, out_filters, bn=True):
            block = [
                nn.Conv1d(in_filters, out_filters, 4, 2, 1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout(0.25)
            ]
            if bn:
                block.append(nn.BatchNorm1d(out_filters, 0.8))
            return block

        self.conv_blocks = nn.Sequential(
            # nn.AdaptiveAvgPool1d(30),
            *conv_block(code_shape[0], 16, bn=False),
            *conv_block(16, 32),
            # *conv_block(32, 64),
            # *conv_block(64, 128),
        )
        # (2, 8) -> (16, 4) -> (32, 2)

        # height and width of downsampled code
        out_shape = 2  # code_shape[1] // 2 ** 2

        # output layers (flatten)
        self.linear_layer1 = nn.Linear(32 * out_shape, 1)  # (B, 64) -> (B, 1)
        self.linear_layer2 = nn.Sequential(nn.Linear(32 * out_shape, 2))  # nn.Softmax())

    def reparameterization(self, mu, logvar):
        Tensor = torch.cuda.FloatTensor if mu.is_cuda else torch.FloatTensor
        std = torch.exp(logvar / 2)
        sampled_z = Variable(Tensor(np.random.normal(0, 1, (mu.size(0), 1))))
        z = sampled_z * std + mu
        return z

    def forward(self, range_code):
        if self.conv_type != 1:  # (2, 8, 8) -> (2, 8)
            # range_code = range_code[:, :, :, 0].squeeze() if self.expand else range_code.squeeze()
            range_code = range_code[:, :, 0].view(range_code.size(0), range_code.size(1), -1)
        code_map = self.conv_blocks(range_code)  # (32, 2)
        err_est = code_map.view(code_map.shape[0], -1)  # flatten for outputs
        
        if self.soft:
            err_est = self.linear_layer2(err_est)  # (B, 64) -> (B, 2)
            mu = err_est[:, 0]
            logvar = err_est[:, 1]
            err_est = self.reparameterization(mu, logvar)
        else:
            err_est = self.linear_layer1(err_est)  # (B, 64) -> (B, 1)

        return err_est


class RestorerConv2d(nn.Module):
    def __init__(self, code_shape, soft=False, filters=64, conv_type=1, expand=False):
        super(RestorerConv2d, self).__init__()
        self.soft = soft
        self.conv_type = conv_type
        self.expand = expand

        def conv_block(in_filters, out_filters, bn=True):
            block = [
                nn.Conv2d(in_filters, out_filters, 4, 2, 1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout(0.25)
            ]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.conv_blocks = nn.Sequential(
            *conv_block(code_shape[0], 16, bn=False),
            *conv_block(16, 32),
            # *conv_block(32, 64),
            # *conv_block(64, 128),
        )
        # 2d: (2, 8, 8) -> (16, 4, 4) -> (32, 2, 2)

        # height and width of downsampled code
        out_shape = (2, 2)  # code_shape[1] // 2 ** 2

        # output layers (flatten)
        self.conv_layer1 = nn.Linear(32 * 2 * 2, 1)
        self.conv_layer2 = nn.Sequential(nn.Linear(32 * 2 * 2, 2))  # nn.Softmax())

    def reparameterization(self, mu, logvar):
        Tensor = torch.cuda.FloatTensor if mu.is_cuda else torch.FloatTensor
        std = torch.exp(logvar / 2)
        sampled_z = Variable(Tensor(np.random.normal(0, 1, (mu.size(0), 1))))
        z = sampled_z * std + mu
        return z

    def forward(self, range_code):
        if self.conv_type == 1:  # (2, 8) -> (2, 8, 8)
            range_code = range_code.unsqueeze(3).expand((range_code.size(0), range_code.size(1), range_code.size(2), range_code.size(2)))
        elif not self.expand:  # (2, 8, 1) -> (2, 8, 8)
            range_code = range_code.expand((range_code.size(0), range_code.size(1), range_code.size(2), range_code.size(2)))
            
        code_map = self.conv_blocks(range_code)  # (2, 8, 8) -> (32, 2, 2)
        err_est = code_map.view(code_map.shape[0], -1)  # flatten for outputs
        # note that globalpooling is not a good choice since omit dims

        if self.soft:
            err_est = self.conv_layer2(err_est)  # (B, 128) -> (B, 2)
            mu = err_est[:, 0]
            logvar = err_est[:, 1]
            err_est = self.reparameterization(mu, logvar)
        else:
            err_est = self.conv_layer1(err_est)  # (B, 128) -> (B, 1)

        return err_est


class RestorerConv2dNoExpand(nn.Module):
    def __init__(self, code_shape, soft=False, filters=64):
        super(RestorerConv2dNoExpand, self).__init__()
        self.soft = soft

        def conv_block(in_filters, out_filters, bn=True):
            block = [
                nn.Conv2d(in_filters, out_filters, (4, 1), (2, 1), (1, 0)),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout(0.25)
            ]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.conv_blocks = nn.Sequential(
            nn.AdaptiveAvgPool2d((32, 1)),
            *conv_block(code_shape.size(0), 16, bn=False),
            *conv_block(16, 32),
            *conv_block(32, 64),
            *conv_block(64, 128),
        )
        # 2d_no_expand: (256, 38/39, 1) -> (256, 32, 1) -> (16, 16, 1) -> (128, 2, 1)

        # height and width of downsampled code
        out_shape = 2  # code_shape // 2 ** 4

        # output layers (flatten)
        self.conv_layer1 = nn.Sequential(nn.Linear(128 * out_shape, 1))
        self.conv_layer2 = nn.Sequential(nn.Linear(128 * out_shape, 2))  # nn.Softmax())

    def reparameterization(self, mu, logvar):
        Tensor = torch.cuda.FloatTensor if mu.is_cuda else torch.FloatTensor
        std = torch.exp(logvar / 2)
        sampled_z = Variable(Tensor(np.random.normal(0, 1, (mu.size(0), 1))))
        z = sampled_z * std + mu
        return z

    def forward(self, range_code):
        code_map = self.conv_blocks(range_code)  # (128, 2, 1)
        err_est = code_map.view(code_map.shape[0], -1)  # flatten for outputs
        # note that globalpooling is not a good choice since omit dims

        if self.soft:
            err_est = self.conv_layer2(err_est)
            mu = err_est[:, 0]
            logvar = err_est[:, 1]
            err_est = self.reparameterization(mu, logvar)
        else:
            err_est = self.conv_layer1(err_est)

        return err_est
    

# ==================================================
#        Classifiers (env_code->logit)
# ==================================================


class ClassifierLinear(nn.Module):
    """Constrain env code with environment labels."""
    def __init__(self, env_dim, num_classes, filters=16):
        super(ClassifierLinear, self).__init__()

        self.env_dim = env_dim  # // 2 fot rv
        self.num_classes = num_classes

        self.layers = nn.Sequential(
            nn.Linear(self.env_dim, filters),
            nn.LeakyReLU(),
            nn.Linear(filters, filters*2),
            nn.LeakyReLU(),
            nn.Linear(filters*2, filters),
            nn.LeakyReLU(),
            nn.Linear(filters, self.num_classes),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Sigmoid()
        )

    def forward(self, env_code):
        code_flat = env_code.view(env_code.size(0), -1)  # (B, env_dim)
        logit = self.layers(code_flat)

        return logit


class ClassifierConv1d(nn.Module):
    """Constrain env code with environment labels."""
    def __init__(self, env_dim, num_classes, filters=16):
        super(ClassifierConv1d, self).__init__()

        self.env_dim = env_dim  # // 2 fot rv
        self.num_classes = num_classes

        def conv_block(in_filters, out_filters, bn=True):
            block = [
                nn.Conv1d(in_filters, out_filters, 1, 1, 0),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout(0.25)
            ]
            if bn:
                block.append(nn.BatchNorm1d(out_filters, 0.8))
            return block
        
        self.conv_blocks = nn.Sequential(
            *conv_block(self.env_dim, filters, bn=False),
            *conv_block(filters, filters),
        )
        # (b, 8, 1) -> (b, 16, 1) -> (b, 16, 1)

        self.linear = nn.Sequential(
            nn.Linear(filters, self.num_classes),
            nn.LeakyReLU(0.2, inplace=True),
        )
        # (b, filters) -> (b, num_classes)

    def forward(self, env_code):
        code_flat = env_code.view(env_code.size(0), -1)  # (B, env_dim)
        code_input = code_flat.unsqueeze(2)  # (B, env_dim, 1)
        logit = self.conv_blocks(code_input)  # (B, filters, 1)
        logit_flat = logit.view(logit.size(0), -1)  # (B, filters)
        logit_output = self.linear(logit_flat)  # (B, num_classes)

        return logit_output


class ClassifierConv2d(nn.Module):
    """Constrain env code with environment labels."""
    def __init__(self, env_dim, num_classes, filters=16):
        super(ClassifierConv2d, self).__init__()

        self.env_dim = env_dim  # // 2 fot rv
        self.num_classes = num_classes

        def conv_block(in_filters, out_filters, bn=True):
            block = [
                nn.Conv2d(in_filters, out_filters, 1, 1, 0),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout(0.25)
            ]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block
        
        self.conv_blocks = nn.Sequential(
            *conv_block(self.env_dim, filters, bn=False),
            *conv_block(filters, filters),
        )
        # (b, 8, 1, 1) -> (b, 16, 1, 1) -> (b, 16, 1, 1)

        self.linear = nn.Sequential(
            nn.Linear(filters, self.num_classes),
            nn.LeakyReLU(0.2, inplace=True),
        )
        # (b, filters) -> (b, num_classes)

    def forward(self, env_code):
        code_flat = env_code.view(env_code.size(0), -1)  # (B, env_dim)
        code_input = code_flat.unsqueeze(2)
        code_input = code_input.unsqueeze(3)  # (B, env_dim, 1, 1)
        logit = self.conv_blocks(code_input)  # (B, filters, 1, 1)
        logit_flat = logit.view(logit.size(0), -1)  # (B, filters)
        logit_output = self.linear(logit_flat)  # (B, num_classes)

        return logit_output


# ====================================
#         Helper Functions
# ====================================


class MLP(nn.Module):
    """predicts AdaIN parameters"""
    def __init__(self, input_dim, output_dim, dim=256, n_blk=3, activ="relu"):
        super(MLP, self).__init__()
        layers = [nn.Linear(input_dim, dim), nn.ReLU(inplace=True)]
        for _ in range(n_blk - 2):
            layers += [nn.Linear(dim, dim), nn.ReLU(inplace=True)]
        layers += [nn.Linear(dim, output_dim)]
        self.model = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.model(x.view(x.size(0), -1))


class LayerNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super(LayerNorm, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps

        if self.affine:
            self.gamma = nn.Parameter(torch.Tensor(num_features).uniform_())
            self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        shape = [-1] + [1] * (x.dim() - 1)
        mean = x.view(x.size(0), -1).mean(1).view(*shape)
        std = x.view(x.size(0), -1).std(1).view(*shape)
        x = (x - mean) / (std + self.eps)

        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)
            x = x * self.gamma.view(*shape) + self.beta.view(*shape)
        return x


class ResidualBlock1d(nn.Module):
    def __init__(self, features, norm="in"):
        super(ResidualBlock1d, self).__init__()

        norm_layer = AdaptiveInstanceNorm1d if norm == "adain" else nn.InstanceNorm1d

        self.block = nn.Sequential(
            nn.ReflectionPad1d(1),
            nn.Conv1d(features, features, 3),
            norm_layer(features),
            nn.ReLU(inplace=True),
            nn.ReflectionPad1d(1),
            nn.Conv1d(features, features, 3),
            norm_layer(features),
        )

    def forward(self, x):
        return x + self.block(x)

    
class ResidualBlock2d(nn.Module):
    def __init__(self, features, norm="in"):
        super(ResidualBlock2d, self).__init__()

        norm_layer = AdaptiveInstanceNorm2d if norm == "adain" else nn.InstanceNorm2d

        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(features, features, 3),
            norm_layer(features),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(features, features, 3),
            norm_layer(features),
        )

    def forward(self, x):
        return x + self.block(x)


class ResidualBlock2dNoExpand(nn.Module):
    def __init__(self, features, norm="in"):
        super(ResidualBlock2dNoExpand, self).__init__()

        norm_layer = AdaptiveInstanceNorm2d if norm == "adain" else nn.InstanceNorm2d

        self.block = nn.Sequential(
            nn.ReflectionPad2d((1, 0)),
            nn.Conv2d(features, features, kernel_size=(3, 1)),
            norm_layer(features),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d((1, 0)),
            nn.Conv2d(features, features, kernel_size=(3, 1)),
            norm_layer(features),
        )

    def forward(self, x):
        return x + self.block(x)


class AdaptiveInstanceNorm1d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(AdaptiveInstanceNorm1d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        # weight and bias are dynamically assigned
        self.weight = None
        self.bias = None
        # just dummy buffers, not used
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))

    def forward(self, x):
        assert (
            self.weight is not None and self.bias is not None
        ), "Please assign weight and bias before calling AdaIN!"
        b, c, w = x.size()  # b, c, h, w
        running_mean = self.running_mean.repeat(b)
        running_var = self.running_var.repeat(b)

        # Apply instance norm
        x_reshaped = x.contiguous().view(1, b * c, w)

        out = F.batch_norm(
            x_reshaped, running_mean, running_var, self.weight, self.bias, True, self.momentum, self.eps
        )

        return out.view(b, c, w)

    def __repr__(self):
        return self.__class__.__name__ + "(" + str(self.num_features) + ")"


class AdaptiveInstanceNorm2d(nn.Module):
    """Reference: https://github.com/NVlabs/MUNIT/blob/master/networks.py"""
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(AdaptiveInstanceNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        # weight and bias are dynamically assigned
        self.weight = None
        self.bias = None
        # just dummy buffers, not used
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))

    def forward(self, x):
        assert (
            self.weight is not None and self.bias is not None
        ), "Please assign weight and bias before calling AdaIN!"
        b, c, h, w = x.size()
        running_mean = self.running_mean.repeat(b)
        running_var = self.running_var.repeat(b)

        # Apply instance norm
        x_reshaped = x.contiguous().view(1, b * c, h, w)

        out = F.batch_norm(
            x_reshaped, running_mean, running_var, self.weight, self.bias, True, self.momentum, self.eps
        )
        return out.view(b, c, h, w)

    def __repr__(self):
        return self.__class__.__name__ + "(" + str(self.num_features) + ")"
