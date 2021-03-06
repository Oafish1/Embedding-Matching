import numpy as np
import torch
import torch.nn as nn


class EmbeddingGenerationModel(nn.Module):
    """Thin, simple embedding generation model"""
    def __init__(self, num_mentees, num_mentors, dim=10):
        super().__init__()

        self.mentees = nn.Embedding(num_mentees, dim)
        self.mentors = nn.Embedding(num_mentors, dim)

    def forward(self, e_id, o_id, manual_e=False):
        """Cosine similarity between mentee ``e_id`` and mentor ``o_id``"""
        e_vec = self.mentees(e_id) if not manual_e else e_id
        o_vec = self.mentors(o_id)
        e_norm = torch.norm(e_vec, dim=1)
        o_norm = torch.norm(o_vec, dim=1)
        return torch.sum(o_vec * e_vec, dim=1) / (o_norm * e_norm)


class EmbeddingInferenceModel(nn.Module):
    """Thin, simple embedding inference model"""
    def __init__(self, input_features, dim=10):
        super().__init__()

        self.layers = nn.Sequential(
            # nn.Linear(input_features, 2*input_features),
            # nn.BatchNorm1d(2*input_features),
            # nn.LeakyReLU(.1, True),
            #
            # nn.Linear(2*input_features, 2*input_features),
            # nn.BatchNorm1d(2*input_features),
            # nn.LeakyReLU(.1, True),
            #
            # nn.Linear(2*input_features, input_features),
            # nn.BatchNorm1d(input_features),
            # nn.LeakyReLU(.1, True),

            nn.Linear(input_features, dim),
            nn.BatchNorm1d(dim),
        )


    def forward(self, x):
        """Forward pass for the model"""
        return self.layers(x)


class LatentModel(nn.Module):
    """Thin autoencoder for automated feature generation"""
    def __init__(self, input_dim, latent_dim=10):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 2*input_dim),
            nn.BatchNorm1d(2*input_dim),
            nn.LeakyReLU(),

            nn.Linear(2*input_dim, 2*input_dim),
            nn.BatchNorm1d(2*input_dim),
            nn.LeakyReLU(),

            nn.Linear(2*input_dim, input_dim),
            nn.BatchNorm1d(input_dim),
            nn.LeakyReLU(),

            nn.Linear(input_dim, latent_dim),
            nn.BatchNorm1d(latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, input_dim),
            nn.BatchNorm1d(input_dim),
            nn.LeakyReLU(),

            nn.Linear(input_dim, 2*input_dim),
            nn.BatchNorm1d(2*input_dim),
            nn.LeakyReLU(),

            nn.Linear(2*input_dim, 2*input_dim),
            nn.BatchNorm1d(2*input_dim),
            nn.LeakyReLU(),

            nn.Linear(2*input_dim, input_dim),
            nn.BatchNorm1d(input_dim),
        )

    def forward(self, x):
        """Generate latent space and reconstruction from autoencoder model"""
        latent = self.encoder(x)
        reconstruction = self.decoder(latent)
        return latent, reconstruction
