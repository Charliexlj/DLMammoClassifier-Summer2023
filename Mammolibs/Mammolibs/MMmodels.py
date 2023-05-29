import torch
import torch.nn as nn


def conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )


class Encoder(nn.Module):
    def __init__(self, in_channel=1, base_channel=32):
        super(Encoder, self).__init__()
        # 256*256*1
        n = base_channel
        self.conv1 = conv_block(1, n)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 128*128*32
        n = n*2
        self.conv2 = conv_block(n//2, n)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 64*64*64
        n = n*2
        self.conv3 = conv_block(n//2, n)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 32*32*128
        n = n*2
        self.conv4 = conv_block(n//2, n)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 16*16*256
        n = n*2
        self.conv5 = conv_block(n//2, n)
        # 16*16*512

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(self.pool1(x))
        x = self.conv3(self.pool2(x))
        x = self.conv4(self.pool3(x))
        x = self.conv5(self.pool4(x))
        return x


class Decoder(nn.Module):
    def __init__(self, in_channel=512, out_channel=2, mode="Autoencoder"):
        super(Decoder, self).__init__()
        # 16*16*512
        n = in_channel//2
        self.upconv4 = nn.ConvTranspose2d(n*2, n, kernel_size=2, stride=2)
        # 16*16*256
        if mode == "Autoencoder": self.conv4 = conv_block(n, n) # noqa
        else: self.conv4 = conv_block(n*2, n) # noqa
        # 32*32*128
        n = n//2
        self.upconv3 = nn.ConvTranspose2d(n*2, n, kernel_size=2, stride=2)
        if mode == "Autoencoder": self.conv3 = conv_block(n, n) # noqa
        else: self.conv3 = conv_block(n*2, n) # noqa
        # 64*64*64
        n = n//2
        self.upconv2 = nn.ConvTranspose2d(n*2, n, kernel_size=2, stride=2)
        if mode == "Autoencoder": self.conv2 = conv_block(n, n) # noqa
        else: self.conv2 = conv_block(n*2, n) # noqa
        # 128*128*32
        n = n//2
        self.upconv1 = nn.ConvTranspose2d(n*2, n, kernel_size=2, stride=2)
        if mode == "Autoencoder": self.conv1 = conv_block(n, out_channel) # noqa
        else: self.conv1 = conv_block(n*2, out_channel) # noqa
        # 256*256*2
        self.conv = nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=1) # noqa

    def forward(self, x):
        x = self.conv4(self.upconv4(x))
        x = self.conv3(self.upconv3(x))
        x = self.conv2(self.upconv2(x))
        x = self.conv1(self.upconv1(x))
        x = self.conv(x)
        return x


class Autoencoder(nn.Module):
    def __init__(self, in_channel=1, out_channel=2, num_filter=32):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(in_channel=in_channel, base_channel=32)
        self.decoder = Decoder(out_channel=2, mode="Autoencoder")

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return torch.sigmoid(x)


class UNet(nn.Module):
    def __init__(self, in_channel=1, out_channel=2, num_filter=32):
        super(UNet, self).__init__()
        self.encoder = Encoder(in_channel=in_channel, n=32)
        self.decoder = Decoder(in_channel=512, out_channel=out_channel, mode="UNet") # noqa

    def forward(self, x):
        enc1 = self.encoder.conv1(x)
        enc2 = self.encoder.conv2(self.encoder.pool1(enc1))
        enc3 = self.encoder.conv3(self.encoder.pool2(enc2))
        enc4 = self.encoder.conv4(self.encoder.pool3(enc3))

        bottleneck = self.encoder.conv5(self.encoder.pool4(enc4))

        dec4 = self.decoder.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder.conv4(dec4)
        dec3 = self.decoder.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder.conv3(dec3)
        dec2 = self.decoder.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder.conv2(dec2)
        dec1 = self.decoder.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder.conv1(dec1)
        return torch.sigmoid(self.decoder.conv(dec1))


class Pretrain_Encoder(nn.Module):
    def __init__(self, in_channel=1, out_vector=1024, num_filter=32):
        super(Pretrain_Encoder, self).__init__()
        self.encoder = Encoder(
            in_channel=in_channel, base_channel=num_filter)
        self.fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(16*16*512, 2048),
            nn.ReLU()
            )
        self.fc2 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(2048, 2048),
            nn.ReLU()
            )
        self.fc3 = nn.Linear(2048, out_vector)

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


def NT_Xent_loss(a, b):
    tau = 1
    a_norm = torch.norm(a, dim=1).reshape(-1, 1)
    a_cap = torch.div(a, a_norm)
    b_norm = torch.norm(b, dim=1).reshape(-1, 1)
    b_cap = torch.div(b, b_norm)
    a_cap_b_cap = torch.cat([a_cap, b_cap], dim=0)
    a_cap_b_cap_transpose = torch.t(a_cap_b_cap)
    b_cap_a_cap = torch.cat([b_cap, a_cap], dim=0)
    sim = torch.mm(a_cap_b_cap, a_cap_b_cap_transpose)
    sim_by_tau = torch.div(sim, tau)
    exp_sim_by_tau = torch.exp(sim_by_tau)
    sum_of_rows = torch.sum(exp_sim_by_tau, dim=1)
    exp_sim_by_tau_diag = torch.diag(exp_sim_by_tau)
    numerators = torch.exp(torch.div(torch.nn.CosineSimilarity()
                           (a_cap_b_cap, b_cap_a_cap), tau))
    denominators = sum_of_rows - exp_sim_by_tau_diag
    num_by_den = torch.div(numerators, denominators)
    neglog_num_by_den = -torch.log(num_by_den)
    return torch.mean(neglog_num_by_den)
