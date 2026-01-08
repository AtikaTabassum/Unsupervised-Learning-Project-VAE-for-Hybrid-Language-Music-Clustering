import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiModalConvVAE(nn.Module):
    def __init__(self, lyrics_dim=384, latent_dim=32):
        super().__init__()
        
        self.enc_conv1 = nn.Conv2d(1, 16, 3, stride=2, padding=1)
        self.enc_conv2 = nn.Conv2d(16, 32, 3, stride=2, padding=1)
        self.enc_conv3 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.flatten_dim = 64 * 8 * 8

        self.fc_lyrics = nn.Linear(lyrics_dim, 128)

        self.fc1 = nn.Linear(self.flatten_dim + 128, 256)
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)

        self.fc_dec = nn.Linear(latent_dim, self.flatten_dim)
        self.dec_conv1 = nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1)
        self.dec_conv2 = nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1)
        self.dec_conv3 = nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1)

    def encode(self, x, lyrics_emb):
        x = F.relu(self.enc_conv1(x))
        x = F.relu(self.enc_conv2(x))
        x = F.relu(self.enc_conv3(x))
        x = x.view(x.size(0), -1)  # flatten
        lyrics_feat = F.relu(self.fc_lyrics(lyrics_emb))
        combined = torch.cat([x, lyrics_feat], dim=1)
        h = F.relu(self.fc1(combined))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        x = F.relu(self.fc_dec(z))
        x = x.view(-1, 64, 8, 8)
        x = F.relu(self.dec_conv1(x))
        x = F.relu(self.dec_conv2(x))
        x = torch.sigmoid(self.dec_conv3(x))
        return x

    def forward(self, x, lyrics_emb):
        mu, logvar = self.encode(x, lyrics_emb)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar
