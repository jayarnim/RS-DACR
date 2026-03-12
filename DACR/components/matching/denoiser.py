import torch
import torch.nn as nn


class DenoisingLayer(nn.Module):
    def __init__(
        self,
        dim: int,
    ):
        super().__init__()

        # global attr
        self.dim = dim

        # generate layers
        self._set_up_components()

    def forward(
        self, 
        noisy_vec: torch.Tensor, 
    ):
        weights = self.transform(noisy_vec)
        denoised_vec = weights * noisy_vec
        return denoised_vec

    def _set_up_components(self):
        self._create_layers()

    def _create_layers(self):
        self.transform = nn.Sequential(
            nn.Linear(self.dim, self.dim),
            nn.Softmax(dim=-1),
        )