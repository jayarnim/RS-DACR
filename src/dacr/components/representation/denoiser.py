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
        X: torch.Tensor, 
    ):
        weights = self.transform(X)
        X_denoised = weights * X
        return X_denoised

    def _set_up_components(self):
        self._create_layers()

    def _create_layers(self):
        self.transform = nn.Sequential(
            nn.Linear(self.dim, self.dim),
            nn.Softmax(dim=-1),
        )