import numpy as np
import torch
import torch.nn as nn


class Model(nn.Module):
    """Thin, simple NN model"""
    def __init__(self, num_mentors, num_mentees):
        super().__init__()

        self.mentors = nn.Embedding(num_mentors, 10)
        self.mentees = nn.Embedding(num_mentees, 10)

    def forward(self, o_id, e_id):
        """Forward pass for the model"""
        o_vec = self.mentors(o_id)
        e_vec = self.mentees(e_id)
        o_norm = torch.norm(o_vec, dim=1)
        e_norm = torch.norm(e_vec, dim=1)
        return torch.sum(o_vec * e_vec, dim=1) / (o_norm * e_norm)
