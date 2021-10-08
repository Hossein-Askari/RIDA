import torch
from torch import nn
from myargs import args


class GaussianNoise(nn.Module):
    def __init__(self, sigma=1.0):
        super().__init__()
        self.sigma = sigma
        self.noise = torch.tensor(0.0).cuda()

    def forward(self, x):
        if self.training:
            sampled_noise = self.noise.repeat(*x.size()).normal_(mean=0, std=self.sigma)
            x = x + sampled_noise
        return x

class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()

        self.n_features = args.feature_size
        self.input_size = args.image_size
        self.nc = 3

        self.instancenorm = nn.InstanceNorm2d(3, momentum=1, eps=1e-3)  # L-17

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, self.n_features, 3, 1, 1),  # L-16
            nn.BatchNorm2d(self.n_features, momentum=0.99, eps=1e-3),  # L-16
            nn.LeakyReLU(negative_slope=0.1, inplace=True),  # L-16
            nn.Conv2d(self.n_features, self.n_features, 3, 1, 1),  # L-15
            nn.BatchNorm2d(self.n_features, momentum=0.99, eps=1e-3),  # L-15
            nn.LeakyReLU(negative_slope=0.1, inplace=True),  # L-15
            nn.Conv2d(self.n_features, self.n_features, 3, 1, 1),  # L-14
            nn.BatchNorm2d(self.n_features, momentum=0.99, eps=1e-3),  # L-14
            nn.LeakyReLU(negative_slope=0.1, inplace=True),  # L-14
            nn.MaxPool2d(2),  # L-13
            nn.Dropout(0.5),  # L-12
            GaussianNoise(args.gaussian_noise),  # L-11
            nn.Conv2d(self.n_features, self.n_features, 3, 1, 1),  # L-10
            nn.BatchNorm2d(self.n_features, momentum=0.99, eps=1e-3),  # L-10
            nn.LeakyReLU(negative_slope=0.1, inplace=True),  # L-10
            nn.Conv2d(self.n_features, self.n_features, 3, 1, 1),  # L-9
            nn.BatchNorm2d(self.n_features, momentum=0.99, eps=1e-3),  # L-9
            nn.LeakyReLU(negative_slope=0.1, inplace=True),  # L-9
            nn.Conv2d(self.n_features, self.n_features, 3, 1, 1),  # L-8
            nn.BatchNorm2d(self.n_features, momentum=0.99, eps=1e-3),  # L-8
            nn.LeakyReLU(negative_slope=0.1, inplace=True),  # L-8
            nn.MaxPool2d(2),  # L-7
            nn.Dropout(0.5),  # L-6
            GaussianNoise(args.gaussian_noise),  # L-5
        )

        self.encode_fc = nn.Sequential(
            nn.Flatten(),
            # nn.Linear(self.n_features * (self.input_size // 4) * (self.input_size // 4) , self.n_features * (self.input_size // 4) * (self.input_size // 4))
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                m.track_running_stats = False

    def track_bn_stats(self, track):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.track_running_stats = track

    def forward(self, x, track_bn=False):
        if track_bn:
            self.track_bn_stats(True)
        x = self.instancenorm(x)
        features = self.feature_extractor(x)
        features = self.encode_fc(features)

        if track_bn:
            self.track_bn_stats(False)

        return features

class Decoder(nn.Module):
    def __init__(self, args):
        super(Decoder, self).__init__()

        self.n_features = args.feature_size
        self.input_size = args.image_size
        self.nc = 3

        self.decode_fc = nn.Sequential(
            nn.Linear(args.num_latent, self.n_features * (self.input_size // 4) * (self.input_size // 4)),
            nn.BatchNorm1d(self.n_features * (self.input_size // 4) * (self.input_size // 4)),
            nn.ReLU(True),
        )

        self.deconv = nn.Sequential(
            
            nn.ConvTranspose2d(self.n_features, self.n_features, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.n_features, momentum=0.99, eps=1e-3),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),    
            nn.ConvTranspose2d(self.n_features, self.n_features, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.n_features, momentum=0.99, eps=1e-3),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.ConvTranspose2d(self.n_features, self.n_features, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.n_features, momentum=0.99, eps=1e-3),
            nn.LeakyReLU(negative_slope=0.1, inplace=True), 
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ConvTranspose2d(self.n_features, self.n_features, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.n_features, momentum=0.99, eps=1e-3),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.ConvTranspose2d(self.n_features, self.n_features, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.n_features, momentum=0.99, eps=1e-3),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.ConvTranspose2d(self.n_features, self.nc, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.nc, momentum=0.99, eps=1e-3),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Tanh()
        )        

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                m.track_running_stats = False

    def track_bn_stats(self, track):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.track_running_stats = track

    def forward(self, x, track_bn=False):
        if track_bn:
            self.track_bn_stats(True)

        deconv_input = self.decode_fc(x)
        deconv_input = deconv_input.view(-1, self.n_features, self.input_size//4, self.input_size//4)
        decoded = self.deconv(deconv_input)

        if track_bn:
            self.track_bn_stats(False)

        return decoded

class Classifier(nn.Module):
    def __init__(self, args):
        super(Classifier, self).__init__()

        self.n_features = args.feature_size
        self.nc = 3
        self.args = args
        self.input_size = args.image_size

        self.classifier = nn.Sequential(
            nn.Conv2d(self.n_features, self.n_features, 3, 1, 1),  # L-4
            nn.BatchNorm2d(self.n_features, momentum=0.99, eps=1e-3),  # L-4
            nn.LeakyReLU(negative_slope=0.1, inplace=True),  # L-4
            nn.Conv2d(self.n_features, self.n_features, 3, 1, 1),  # L-3
            nn.BatchNorm2d(self.n_features, momentum=0.99, eps=1e-3),  # L-3
            nn.LeakyReLU(negative_slope=0.1, inplace=True),  # L-3
            nn.Conv2d(self.n_features, self.n_features, 3, 1, 1),  # L-2
            nn.BatchNorm2d(self.n_features, momentum=0.99, eps=1e-3),  # L-2
            nn.LeakyReLU(negative_slope=0.1, inplace=True),  # L-2
            nn.AdaptiveAvgPool2d(1),  # L-1
            nn.Conv2d(self.n_features, args.classes, 1)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                m.track_running_stats = False

    def track_bn_stats(self, track):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.track_running_stats = track

    def forward(self, x, track_bn=False):
        if track_bn:
            self.track_bn_stats(True)

        x = x.view(-1,self.n_features, self.input_size//4, self.input_size//4)
        logits = self.classifier(x) 
        
        if track_bn:
            self.track_bn_stats(False)
        
        return logits.view(x.size(0), 9)

class Discriminator(nn.Module):
    def __init__(self, large=False):
        super(Discriminator, self).__init__()

        n_features = 192 if large else 64

        self.disc = nn.Sequential(
            nn.Linear(n_features * 1 * 8 * 8, 100),
            nn.ReLU(True),
            nn.Linear(100, 1)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.disc(x).view(x.size(0), -1)

class EMA:
    def __init__(self, decay):
        self.decay = decay
        self.shadow = {}

    def register(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
        self.params = self.shadow.keys()

    def __call__(self, model):
        if self.decay > 0:
            for name, param in model.named_parameters():
                if name in self.params and param.requires_grad:
                    self.shadow[name] -= (1 - self.decay) * (self.shadow[name] - param.data)
                    param.data = self.shadow[name]
