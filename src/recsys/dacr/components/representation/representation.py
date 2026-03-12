import torch
import torch.nn as nn
from .combination import Concatenation
from .denoiser import DenoisingLayer


class RepresentationLayer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: list,
        dropout: float,
    ):
        super().__init__()

        # global attr
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout

        # generate layers
        self._set_up_components()

    def forward(
        self, 
        user_emb: torch.Tensor, 
        item_emb: torch.Tensor,
    ):
        user_rep_slice = self._representation(user_emb, "user")
        item_rep_slice = self._representation(item_emb, "item")
        return user_rep_slice, item_rep_slice

    def _representation(self, emb, entity):
        denoised = self.denoiser[entity](emb)
        combined = self.combination[entity](emb, denoised)
        return self.mlp[entity](combined)

    def _set_up_components(self):
        self._create_components()
        self._create_layers()

    def _create_components(self):
        kwargs = dict(
            dim=self.input_dim//2,
        )
        components = dict(
            user=DenoisingLayer(**kwargs),
            item=DenoisingLayer(**kwargs),
        )
        self.denoiser = nn.ModuleDict(components)

        components = dict(
            user=Concatenation(),
            item=Concatenation(),
        )
        self.combination = nn.ModuleDict(components)

    def _create_layers(self):
        components = list(self._yield_linear_block(self.hidden_dim))
        mlp_user = nn.Sequential(*components)

        components = list(self._yield_linear_block(self.hidden_dim))
        mlp_item = nn.Sequential(*components)

        components = dict(
            user=mlp_user,
            item=mlp_item,
        )
        self.mlp = nn.ModuleDict(components)

    def _yield_linear_block(self, hidden_dim):
        IN_FEATRUES = self.input_dim
        
        for OUT_FEATURES in hidden_dim:
            yield nn.Sequential(
                nn.Linear(IN_FEATRUES, OUT_FEATURES),
                nn.LayerNorm(OUT_FEATURES),
                nn.ReLU(),
                nn.Dropout(self.dropout),
            )
            IN_FEATRUES = OUT_FEATURES