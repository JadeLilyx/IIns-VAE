import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


def weights_init_normal(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Conv1d) or isinstance(m, nn.conv2d):
        torch.nn.init.normal_()
    elif isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d):
        torch.nn.init.normal_(m.weights.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


class LambdaLR:
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert (n_epochs - decay_start_epoch) > 0, "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch) / (self.n_epochs - self.decay_start_epoch)


# ------------------ Basic architecture --------------------

class Encoder(nn.Module):
    def __init__(self, conv_type=1, filters=4, n_residual=3, n_downsample=4, env_dim=8, range_dim=4):
        super(Encoder, self).__init__()
        self.conv_type = conv_type
        self.env_dim = env_dim
        self.range_dim = range_dim
        if self.conv_type == 1:
            self.range_encoder = RangeEncoder1d(filters, n_residual, n_downsample, range_dim)
            self.env_encoder = EnvEncoder1d(filters * 4, n_downsample - 2, env_dim)
        elif self.conv_type == 2:
            self.range_encoder = RangeEncoder2d(filters, n_residual, n_downsample, range_dim)
            self.env_encoder = EnvEncoder2d(filters * 4, n_downsample - 2, env_dim)
        else:
            raise ValueError("Unknown convolution type for autoencoder!")

    def forward(self, x):
        if self.conv_type == 1:
            x = x.view(x.size(0), 1, x.size(1))
        else:
            x = x.view(x.size(0), 1, x.size(1), 1).expand((x.size(0), 1, x.size(1), x.size(1)))

        range_code = self.range_encoder(x)  # (B, 1, cir_len, cir_len/none) -> (B, 2, 8, 8/none)
        env_code, env_code_rv, kl_div = self.env_encoder(x)  # (B, 8, 1, 1/none)
        
        return range_code, env_code, env_code_rv, kl_div

    def sample(self, n):
        return torch.randn(n, self.env_dim)


class Decoder(nn.Module):
    def __init__(self, conv_type=1, filters=4, n_residual=3, n_upsample=4, env_dim=8, range_dim=4, out_dim=152):
        super(Decoder, self).__init__()

        self.conv_type = conv_type
        if conv_type == 1:
            self.decoder = Decoder1d(filters, n_residual, n_upsample, range_dim, env_dim, out_dim)
        elif conv_type == 2:
            self.decoder = Decoder2d(filters, n_residual, n_upsample, range_dim, env_dim, out_dim)
        else:
            raise ValueError("Unknown convolutional type for autoencoder!")

    def forward(self, range_code, env_code):
        x_recon = self.decoder(range_code, env_code)  # (B, 2, 8, 8/none) + (B, 8, 1, 1/none) -> (B, 1, 152, 152/none)

        if self.conv_type == 1:
                x_recon = x_recon.squeeze()  # (B, cir_len)
        else:
            x_recon = x_recon[:, :, :, 0].squeeze()
        
        return x_recon


class Restorer(nn.Module):
    """Constrain range code with range error."""
    def __init__(self, use_soft=False, layer_type=1, conv_type=1, range_dim=2, n_downsample=4):
        super(Restorer, self).__init__()

        self.use_soft = use_soft
        
        if conv_type == 1:  # (2, 8)
            self.code_shape = (range_dim, 128 // (2 ** n_downsample))
        elif conv_type == 2:  # (2, 8, 8)
            self.code_shape = (range_dim, 128 // (2 ** n_downsample), 128 // (2 ** n_downsample))

        if layer_type == 1:  # Linear
            self.restorer = RestorerLinear(self.code_shape, use_soft)
        elif layer_type == 2:  # Conv1d
            self.restorer = RestorerConv1d(self.code_shape, use_soft)
        elif layer_type == 3:  # Conv2d
            self.restorer = RestorerConv2d(self.code_shape, use_soft)
        else:
            raise ValueError("Unknown layer type for restorer network!")

    def forward(self, range_code):
        err_est = self.restorer(range_code)
        return err_est


class Classifier(nn.Module):
    """Constrain env code with environment label."""
    def __init__(self, env_dim, num_classes, filters=16, layer_type=1):
        super(Classifier, self).__init__()

        if layer_type == 1:  # Linear
            self.classifier = ClassifierLinear(env_dim, num_classes, filters)
        elif layer_type == 2:  # Conv1d
            self.classifier = ClassifierConv1d(env_dim, num_classes, filters)
        elif layer_type == 3:  # Conv2d
            self.classifier = ClassifierConv2d(env_dim, num_classes, filters)
        else:
            raise ValueError("Unknown layer type for classifier network!")

    def forward(self, env_code_rv):
        label_est = self.classifier(env_code_rv)
        return label_est


# -------------- Encoders (x -> codes) ---------------


class RangeEncoder1d(nn.Module):
    def __init__(self, filters=4, n_residual=3, n_downsample=4, range_dim=2):
        super(RangeEncoder1d, self).__init__()

        # Initialize input to 128 dim, (1, 152) -> (1, 128)
        layers = []
        layers += [nn.AdaptiveAvgPool1d(128)]  # or input cir_len and use linear layer

        # Iinitial convolution block, (1, 128) -> (4, 128)
        layers += [
            nn.ReflectionPad1d(3),  # (1, 134), 128_6
            nn.Conv1d(1, filters, 7),  # (4, 128), 134-7+1
            nn.InstanceNorm1d(filters),
            nn.ReLU(inplace=True),
        ]

        # Downsampling, (4, 128) -> (8, 64) -> (16, 32) -> (32, 16) -> (64, 8)
        for _ in range(n_downsample):
            layers += [
                nn.Conv1d(filters, filters * 2, 4, stride=2, padding=1),
                nn.InstanceNorm1d(dim * 2),
                nn.ReLU(inplace=True),
            ]
            filters *= 2

        # Residual blocks
        for _ in range(n_residual):
            layers += [ResidualBlock1d(filters, norm="in")]

        # Output layer, (64, 8) -> (range_dim, 8)
        # filters = filters * 2 ** n_downsample  # 64
        layers += [nn.Conv1d(filters, range_code, 1, 1, 0), nn.ReLU(inplace=True)]

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)  # (B, 1, cir_len) -> (B, range_dim, 8)


class RangeEncoder2d(nn.Module):
    def __init__(self, filters=4, n_residual=3, n_downsample=4, range_dim=2):
        super(RangeEncoder2d, self).__init__()

        # Initialize input dim to 128, (1, cir_len, cir_len) -> (1, 128, 128)
        layers = []
        layers += [nn.AdaptiveAvgPool2d(128)]

        # Initial convolution block, (1, 128, 128) -> (4, 128, 128)
        layers += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(1, filters, 7),
            nn.InstanceNorm2d(filters),
            nn.ReLU(inplace=True),
        ]

        # Downsampling
        for _ in range(n_downsample):  # (4, 128, 128) -> (8, 64, 64)
            layers += [
                nn.Conv2d(filters, filters * 2, 4, stride=2, padding=1),
                nn.InstanceNorm2d(filters * 2),
                nn.ReLU(inplace=True),
            ]
            filters *= 2

        # Residual blocks
        for _ in range(n_residual):  # (64, 8, 8)
            layers += [ResidualBlock2d(filters, norm="in")]

        # Output layer, (64, 8, 8) -> (range_dim, 8, 8)
        layers += [nn.Conv2d(filters, range_dim, 1, 1, 0), nn.ReLU(inplace=True)]

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)  # (B, range_dim, 8, 8)


class EnvEncoder1d(nn.Module):
    def __init__(self, filters=16, n_downsample=3, env_dim=8):
        super(EnvEncoder1d, self).__init__()

        self.latent_dim = env_dim // 2

        # Initialize input dim, (1, cir_len) -> (1, 128)
        layers = []
        layers += [nn.AdaptiveAvgPool1d(128)]

        # Initial conv block, (1, 128) -> (16, 128)
        layers += [
            nn.ReflectionPad1d(3),
            nn.Conv1d(1, filters, 7),
            nn.ReLU(inplace=True)
        ]

        # Downsampling, (16, 128) -> (32, 64) -> (64, 32)
        for _ in range(2):
            layers += [
                nn.Conv1d(filters, filters * 2, 4, stride=2, padding=1),
                nn.ReLU(inplace=True)
            ]
            filters *= 2

        # Downsampling with constant depth, (64, 32) -> (64, 16)
        for _ in range(n_downsample - 2):
            layers += [
                nn.Conv1d(filters, filters, 4, stride=2, padding=1),
                nn.ReLU(inplace=True)
            ]

        # Average pool and output layer, (64, 16) -> (64, 1) -> (8, 1)
        layers += [
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(filters, env_dim, 1, 1, 0)
        ]

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        cat = self.model(x)  # (B, env_dim, 1)
        latent_dim = cat.shape[1] // 2  # env_dim/2
        mu, log_sigma = torch.split(cat, latent_dim, dim=1)
        noise = torch.randn_like(mu)
        latent_code = noise * log_sigma.exp() + mu  # (B, env_dim/2, 1)
        kl_div = self._loss(mu, log_sigma)
        
        return cat, latent_code, kl_div

    def _loss(self, mu, log_sigma):
        kl_div = 0.5 * torch.sum((2 * log_sigma).exp() + mu ** 2 - 1 - 2 * log_sigma, dim=1)
        kl_div = kl_div.mean()  # account for batch size

        return kl_div


class EnvEncoder2d(nn.Module):
    def __init__(self, filters=16, n_downsample=3, env_dim=8):
        super(EnvEncoder2d, self).__init__()

        self.latent_dim = env_dim // 2

        # Initialize input dim, (1, cir_len, cir_len) -> (1, 128, 128)
        layers = []
        layers += [nn.AdaptiveAvgPool2d(128)]

        # Initial conv block, (1, 128, 128) -> (16, 128, 128)
        layers += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(1, filters, 7),
            nn.ReLU(inplace=True)
        ]

        # Downsampling, (16, 128, 128) -> (32, 64, 64) -> (64, 32, 32)
        for _ in range(2):
            layers += [
                nn.Conv2d(filters, filters * 2, 4, stride=2, padding=1),
                nn.ReLU(inplace=True)
            ]
            filters *= 2

        # Downsampling with constant depth, (64, 32, 32) -> (64, 16, 16)
        for _ in range(n_downsample - 2):
            layers += [
                nn.Conv2d(filters, filters, 4, stride=2, padding=1),
                nn.ReLU(inplace=True)
            ]

        # Average pool and output layer, (64, 16, 16) -> (64, 1, 1) -> (8, 1, 1)
        layers += [
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(filters, env_dim, 1, 1, 0)
        ]

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        cat = self.model(x)  # (B, env_dim, 1, 1)
        latent_dim = cat.shape[1] // 2
        mu, log_sigma = torch.split(cat, latent_dim, dim=1)
        noise = torch.randn_like(mu)
        latent_code = noise * log_sigma.exp() + mu  # (B, env_dim/2, 1, 1)
        kl_div = self._loss(mu, log_sigma)

        return cat, latent_code, kl_div

    def _loss(self, mu, log_sigma):
        kl_div = 0.5 * torch.sum((2 * log_sigma).exp() + mu ** 2 -1 -2 * log_sigma, dim=1)
        kl_div = kl_div.mean()  # account for batch size

        return kl_div


# ---------------- Encoders (codes -> x_recon) -----------------


class Decoder1d(nn.Module):
    def __init__(self, filters=4, n_residual=3, n_upsample=4, range_dim=4, env_dim=8, out_dim=152):
        super(Decoder1d, self).__init__()

        layers = []
        filters = filters * 2 ** n_upsample  # 64

        # Initialize input, (2, 8) -> (64, 8)
        layers += [
            nn.Conv1d(range_dim, filters, 1, 1, 0),
            nn.ReLU(inplace=True)
        ]

        # Residual blocks, (64, 8)
        for _ in range(n_residual):
            layers += [ResidualBlock1d(filters, norm="adain")]

        # Upsampling, (64, 8) -> (32, 16) -> (16, 32) -> (8, 64) -> (4, 128)
        for _ in range(n_upsample):
            layers += [
                nn.Upsample(scale_factor=2),  # (64, 16)
                nn.Conv1d(filters, filters // 2, 5, stride=1, padding=2)  # (32, 16)
                LayerNorm(filters // 2),
                nn.ReLU(inplace=True)
            ]
            filters = filters // 2

        # Output layer, (4, 128) -> (1, 128)
        layers += [
            nn.ReflectionPad1d(3),
            nn.Conv1d(filters, 1, 7),
            nn.Tanh()
        ]

        layers += [nn.AdaptiveAvgPool1d(out_dim)]  # (1, cir_len)

        self.model = nn.Sequential(*layers)

        # Initiate mlp (predict AdaIN paramters)
        num_adain_params = self.get_num_adain_params()
        self.mlp = MLP(env_dim, num_adain_params)

    def get_num_adain_params(self):
        """Return the number of AdaIN params needed by the model."""
        num_adain_params = 0
        for m in self.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm1d":
                num_adain_params += 2 * m.num_features
        return num_adain_params

    def assign_adain_params(self, adain_params):
        """Assign the adain_params to the AdaIN layers in model."""
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
        self.assign_adain_params(self.mlp(env_code))  # (B, 8, 1) -> (B, num_adain_params, 1)
        # combine range_code in for the final output
        x_recon = self.model(range_code)
        return x_recon


class Decoder2d(nn.Module):
    def __init__(self, filters=4, n_residual=3, n_upsample=4, range_dim=4, env_dim=8, out_dim=152):
        super(Decoder2d, self).__init__()

        layers = []
        filters = filters * 2 ** n_upsample  # 64

        # Initialize input, (2, 8, 8) -> (64, 8, 8)
        layers += [
            nn.Conv2d(range_dim, filters, 1, 1, 0),
            nn.ReLU(inplace=True)
        ]

        # Residual blocks, (64, 8, 8)
        for _ in range(n_residual):
            layers += [ResidualBlock2d(filters, norm="adain")]

        # Upsampling, (64, 8) -> (32, 16) -> (16, 32) -> (8, 64) -> (4, 128)
        for _ in range(n_upsample):
            layers += [
                nn.Upsample(scale_factor=2),  # (64, 16)
                nn.Conv2d(filters, filters // 2, 5, stride=1, padding=2)  # (32, 16)
                LayerNorm(filters // 2),
                nn.ReLU(inplace=True)
            ]
            filters = filters // 2

        # Output layer, (4, 128) -> (1, 128)
        layers += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(filters, 1, 7),
            nn.Tanh()
        ]

        layers += [nn.AdaptiveAvgPool2d(out_dim)]  # (1, cir_len)

        self.model = nn.Sequential(*layers)

        # Initiate mlp (predict AdaIN paramters)
        num_adain_params = self.get_num_adain_params()
        self.mlp = MLP(env_dim, num_adain_params)

    def get_num_adain_params(self):
        """Return the number of AdaIN params needed by the model."""
        num_adain_params = 0
        for m in self.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
                num_adain_params += 2 * m.num_features
        return num_adain_params

    def assign_adain_params(self, adain_params):
        """Assign the adain_params to the AdaIN layers in model."""
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
        # Update AdaIN parameters by MLP prediction based off env_code
        self.assign_adain_params(self.mlp(env_code))  # (B, 8, 1) -> (B, num_adain_params, 1)
        # combine range_code in for the final output
        x_recon = self.model(range_code)
        return x_recon



# ----------------- Restorers (range_code -> err_est) ------------------


class RestorerLinear(nn.Module):
    def __init__(self, code_shape, use_soft=False):
        super(RestorerLinear, self).__init__()

        self.use_soft = use_soft
        self.layers = nn.Sequential(
            # Flatten
            nn.Linear(int(np.prod(code_shape)), 256),  # +1 for dist_gt, 2*8/2*8*8 -> 512
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 64),
            # nn.BatchNorm1d(256, 0.8),  # require batch_size>1
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Linear(256, 2),
        )

        self.linear_layer1 = nn.Linear(64, 1)
        self.linear_layer2 = nn.Linear(64, 2)  # nn.Softmax())
        # self.sigmoid = nn.Sigmoid()  # not 0~1

    def reparameterization(self, mu, logvar):
        Tensor = torch.cuda.FloatTensor if mu.is_cuda else torch.FloatTensor
        std = torch.exp(logvar / 2)
        sampled_z = Variable(Tensor(np.random.normal(0, 1, (mu.size(0), 1))))
        z = sampled_z * std + mu
        return z

    def forward(self, range_code):
        code_flat = range_code.view(range_code.size(0), -1)
        # 1d: (B, 2, 8) -> (B, 16)
        # 2d: (B, 2, 8, 8) -> (B, 128)
        err_est = self.layers(code_flat)
        # 1d: (B, 16) -> (B, 256) -> (B, 128)
        # 2d: (B, 128) -> (B, 256) -> (B, 128)

        if self.soft:
            err_est = self.linear_layer2(err_est)  # 128 -> 2
            mu = err_est[:, 0]
            logvar = err_est[:, 1]
            err_est = self.reparameterization(mu, logvar)  # (B, 2)
        else:
            err_est = self.linear_layer2(err_est)  # 128 -> 1

        return err_est


class RestorerConv1d(nn.Module):
    def __init__(self, code_shape, use_soft=False):
        super(RestorerConv1d, self).__init__()
        
        self.use_soft = use_soft

        def conv_block(in_filters, out_filters, bn=True):
            block = [
                nn.Conv1d(in_filters, out_filters, 4, 2, 1)
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout(0.25)
            ]
            if bn:
                block.append(nn.BatchNorm1d(out_filters, 0.8))
            return blocks
        
        self.init_layer = nn.Linear(np.prod(code_shape), 256)

        self.conv_blocks = nn.Sequential(
            *conv_block(16, 32, bn=False),
            *conv_block(32, 64),
            *conv_block(64, 128)
        )  # (16, 16) -> (32, 8) -> (64, 4) -> (128, 2)

        # height and width of downsampled code
        out_shape = 2  # code_shape[1] // 2 ** 2

        # output layers (flatten)
        self.linear_layer1 = nn.Linear(128 * out_shape, 1)
        self.linear_layer2 = nn.Sequential(nn.Linear(128 * out_shape, 2))  # nn.Softmax()

    def reparameterization(self, mu, logvar):
        Tensor = torch.cuda.FloatTensor if mu.is_cuda else torch.FloatTensor
        std = torch.exp(logvar / 2)
        sampled_z = Variable(Tensor(np.random.normal(0, 1, (mu.size(0), 1))))
        z = sampled_z * std + mu
        return z

    def forward(self, range_code):
        code_flat = range_code.view(range_code.size(0), -1)
        # 1d: (B, 2, 8) -> (B, 16)
        # 2d: (B, 2, 8, 8) -> (B, 128)
        err_init = self.init_layer(code_flat)
        # 1d: (B, 16) -> (B, 256)
        # 2d: (B, 128) -> (B, 256)
        err_in = err_init.view(err_in.size(0), 16, 16)
        err_out = self.conv_blocks(err_in)  # (B, 16, 16) -> (B, 128, 2)
        err_est = err_est.view(err_est.size(0), -1)  # (B, 256)

        if self.soft:
            err_est = self.linear_layer2(err_est)  # 256 -> 2
            mu = err_est[:, 0]
            logvar = err_est[:, 1]
            err_est = self.reparameterization(mu, logvar)  # (B, 2)
        else:
            err_est = self.linear_layer2(err_est)  # 256 -> 1

        return err_est


class RestorerConv2d(nn.Module):
    def __init__(self, code_shape, use_soft=False):
        super(RestorerConv2d, self).__init__()
        
        self.use_soft = use_soft

        def conv_block(in_filters, out_filters, bn=True):
            block = [
                nn.Conv2d(in_filters, out_filters, 4, 2, 1)
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout(0.25)
            ]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return blocks
        
        self.init_layer = nn.Linear(np.prod(code_shape), 512)

        self.conv_blocks = nn.Sequential(
            *conv_block(16, 32, bn=False),
            *conv_block(32, 64),
            *conv_block(64, 128)
        )  # (2, 16, 16) -> (4, 8, 8) -> (8, 4, 4) -> (16, 2, 2)

        # height and width of downsampled code
        out_shape = (2, 2)  # code_shape[1] // 2 ** 2

        # output layers (flatten)
        self.linear_layer1 = nn.Linear(16 * 2 * 2, 1)
        self.linear_layer2 = nn.Sequential(nn.Linear(16 * 2 * 2, 2))  # nn.Softmax()

    def reparameterization(self, mu, logvar):
        Tensor = torch.cuda.FloatTensor if mu.is_cuda else torch.FloatTensor
        std = torch.exp(logvar / 2)
        sampled_z = Variable(Tensor(np.random.normal(0, 1, (mu.size(0), 1))))
        z = sampled_z * std + mu
        return z

    def forward(self, range_code):
        code_flat = range_code.view(range_code.size(0), -1)
        # 1d: (B, 2, 8) -> (B, 16)
        # 2d: (B, 2, 8, 8) -> (B, 128)
        err_init = self.init_layer(code_flat)
        # 1d: (B, 16) -> (B, 512)
        # 2d: (B, 128) -> (B, 512)
        err_in = err_init.view(err_in.size(0), 2, 16, 16)
        err_out = self.conv_blocks(err_in)  # (B, 2, 16, 16) -> (B, 16, 2, 2)
        err_est = err_est.view(err_est.size(0), -1)  # (B, 64)

        if self.soft:
            err_est = self.linear_layer2(err_est)  # 64 -> 2
            mu = err_est[:, 0]
            logvar = err_est[:, 1]
            err_est = self.reparameterization(mu, logvar)  # (B, 2)
        else:
            err_est = self.linear_layer2(err_est)  # 64 -> 1

        return err_est



# ------------------ Classifiers (env_code -> logit) ----------------


class ClassifierLinear(nn.Module):
    """Constrain env code with environment labels."""
    def __init__(self, env_dim, num_classes, filters==16):
        super(ClassifierLinear, self).__init__()

        self.env_dim = env_dim  # // 2 for rv
        self.num_classes = num_classes

        self.layers = nn.Sequential(
            nn.Linear(self.env_dim // 2, filters),
            nn.LeakyReLU(),
            nn.Linear(filters, filters * 2),
            nn.LeakyReLU(),
            nn.Linear(filters * 2, filters),
            nn.LeakyReLU(),
            nn.Linear(filters, self.num_classes),
            nn.LeakyReLU(0.2, inplace=True),
        )  # (4) -> (16) -> (32) -> (16) -> (num_classes)

    def forward(self, env_code_rv):
        code_flat = env_code_rv.view(env_code_rv.size(0), -1)  # (B, env_dim)
        logit = self.layers(code_flat)
        
        return logit


class ClassifierConv1d(nn.Module):
    """Constrain env code with environment labels."""
    def __init__(self, env_dim, num_classes, filters=16):
        super(ClassifierConv1d, self).__init__()

        self.env_dim = env_dim  # // 2 for rv
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
            *conv_block(self.env_dim // 2, filters, bn=False),
            *conv_block(filters, filters * 2),
            *conv_block(filters * 2, filters),
        )
        # (4, 1) -> (16, 1) -> (32, 1) -> (16, 1)

        self.linear = nn.Sequential(
            nn.Linear(filters, self.num_classes),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # (16) -> (num_classes)

    def forward(self, env_code_rv):
        code_flat = env_code_rv.view(env_code_rv.size(0), -1)  # (B, env_dim // 2)
        code_in = code_flat.unsqueeze(2)  # (B, env_dim // 2, 1)
        logit = self.conv_blocks(code_in)  # (B, filters, 1)
        logit_flat = logit.view(logit.size(0), -1)  # (B, filters)
        logit_out = self.linear(logit_flat)  # (B, num_classes)

        return logit_out


class ClassifierConv2d(nn.Module):
    """Constrain env code with environment labels."""
    def __init__(self, env_dim, num_classes, filters=16):
        super(ClassifierConv2d, self).__init__()

        self.env_dim = env_dim
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
            *conv_block(self.env_dim // 2, filters, bn=False),
            *conv_block(filters, filters * 2),
            *conv_block(filters * 2, filters)
        )
        # (4, 1, 1) -> (16, 1, 1) -> (32, 1, 1) -> (16, 1, 1)

        self.linear = nn.Sequential(
            nn.Linear(filters, self.num_classes),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # (16) -> (num_classes)

    def forward(self, env_code_rv):
        code_flat = env_code_rv.view(env_code_rv.size(0), -1)  # (B, env_dim)
        code_in = code_flat.unsqueeze(2)
        code_in = code_in.unsqueeze(3)  # (B, env_dim, 1, 1)
        logit = self.conv_blocks(code_in)  # (B, filters, 1, 1)
        logit_flat = logit.view(logit.size(0), -1)  # (B, filters)
        logit_out = self.linear(logit_flat)  # (B, num_classes)

        return logit_out


# ----------------- Helper Functions -----------------------


class MLP(nn.Module):
    """Predicts AdaIN parameters."""
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

