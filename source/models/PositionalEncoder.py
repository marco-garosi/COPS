import torch
import torch.nn as nn
import math


class PositionalEncoder(nn.Module):
    def __init__(self, input_dim, output_dim, device='cpu'):
        super(PositionalEncoder, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.encoding = self.create_encoding().to(device)

    def create_encoding(self):
        encoding = torch.zeros(self.input_dim, self.output_dim)
        pos = torch.arange(0, self.input_dim, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.output_dim, 2).float() * (-math.log(10000.0) / self.output_dim))
        encoding[:, 0::2] = torch.sin(pos * div_term)
        encoding[:, 1::2] = torch.cos(pos * div_term)

        return encoding

    def forward(self, coordinates):
        # Normalize coordinates
        coordinates = coordinates - coordinates.min(axis=0).values
        coordinates = coordinates / coordinates.max(axis=0).values

        # Assuming coordinates are in the range [0, 1] for simplicity
        # Rescale coordinates to match the positional encoding input dimension
        scaled_coordinates = coordinates * self.input_dim

        # Round coordinates to get integer indices
        indices = scaled_coordinates.long()

        # Clamp indices to fit within the input_dim
        indices = torch.clamp(indices, 0, self.input_dim - 1)

        # Retrieve positional encodings based on indices
        encoded_positions = self.encoding[indices]

        # Sum the positional encodings along the points dimension
        positional_embedding = torch.sum(encoded_positions, dim=1)

        return positional_embedding
