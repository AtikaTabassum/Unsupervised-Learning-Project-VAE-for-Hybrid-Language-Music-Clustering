import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiModalBetaVAE(nn.Module):
    def __init__(self, lyrics_dim=384, latent_dim=32, beta=4.0, num_genres=None, genre_emb_dim=32):
       
        super().__init__()
        self.beta = beta
        self.num_genres = num_genres
        self.genre_emb_dim = genre_emb_dim if num_genres is not None else 0

        self.enc_conv = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.ReLU()
        )

        self.flatten_dim = 128 * 8 * 8

        self.lyrics_fc = nn.Sequential(
            nn.Linear(lyrics_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )

        if num_genres is not None:
            self.genre_emb = nn.Embedding(num_genres, genre_emb_dim)
        else:
            self.genre_emb = None

        
        fusion_dim = self.flatten_dim + 128 + (self.genre_emb_dim if self.genre_emb is not None else 0)
        self.fc_mu = nn.Linear(fusion_dim, latent_dim)
        self.fc_logvar = nn.Linear(fusion_dim, latent_dim)

        
        dec_input_dim = latent_dim + (self.genre_emb_dim if self.genre_emb is not None else 0)
        self.fc_dec = nn.Linear(dec_input_dim, self.flatten_dim)
        self.dec_conv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 4, 2, 1),
            nn.Sigmoid()
        )

    def encode(self, x, lyrics, genre=None):
        x = self.enc_conv(x)
        x = x.view(x.size(0), -1)
        l = self.lyrics_fc(lyrics)
        pieces = [x, l]
        if self.genre_emb is not None:
            if genre is None:
                raise ValueError("Model configured for genre conditioning but `genre` not provided to encode()")
            g = self.genre_emb(genre.long())
            pieces.append(g)
        h = torch.cat(pieces, dim=1)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def decode(self, z, genre=None):
        if self.genre_emb is not None:
            if genre is None:
                raise ValueError("Model configured for genre conditioning but `genre` not provided to decode()")
            g = self.genre_emb(genre.long())
            z = torch.cat([z, g], dim=1)
        x = self.fc_dec(z).view(-1, 128, 8, 8)
        return self.dec_conv(x)

    def forward(self, x, lyrics, genre=None):
        mu, logvar = self.encode(x, lyrics, genre=genre)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, genre=genre)
        return recon, mu, logvar

    def loss(self, recon, x, mu, logvar):
        recon_loss = F.mse_loss(recon, x, reduction="mean")
        kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + self.beta * kl
