import torch
import torch.nn as nn
from . import amlnet, arlnet


class Module(nn.Module):
    def __init__(
        self,
        n_users: int,
        n_items: int,
        n_factors: int,
        hidden_arl: list,
        hidden_aml: list,
        dropout: float,
        interactions: torch.Tensor, 
    ):
        super(Module, self).__init__()
        # attr dictionary for load
        self.init_args = locals().copy()
        del self.init_args["self"]
        del self.init_args["__class__"]

        # device setting
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(DEVICE)

        # global attr
        self.n_users = n_users
        self.n_items = n_items
        self.n_factors = n_factors
        self.hidden_arl = hidden_arl
        self.hidden_aml = hidden_aml
        self.dropout = dropout
        self.interactions = interactions.to(self.device)

        # generate layers
        self._init_layers()

    def forward(
        self, 
        user_idx: torch.Tensor, 
        item_idx: torch.Tensor,
    ):
        """
        user_idx: (B,)
        item_idx: (B,)
        """
        return self.score(user_idx, item_idx)

    def predict(
        self, 
        user_idx: torch.Tensor, 
        item_idx: torch.Tensor,
    ):
        """
        user_idx: (B,)
        item_idx: (B,)
        """
        with torch.no_grad():
            logit = self.score(user_idx, item_idx)
            pred = torch.sigmoid(logit)
        return pred

    def score(self, user_idx, item_idx):
        pred_vector_arl = self.arl_net.arl(user_idx, item_idx)
        pred_vector_aml = self.aml_net.aml(user_idx, item_idx)

        pred_vector = torch.cat(
            tensors=(pred_vector_arl, pred_vector_aml), 
            dim=-1
        )

        logit = self.logit_layer(pred_vector).squeeze(-1)

        return logit

    def _init_layers(self):
        self.arl_net = arlnet.Module(
            n_users=self.n_users,
            n_items=self.n_items,
            hidden=self.hidden_arl,
            dropout=self.dropout,
            interactions=self.interactions,
        )
        self.aml_net = amlnet.Module(
            n_users=self.n_users,
            n_items=self.n_items,
            n_factors=self.n_factors,
            hidden=self.hidden_aml,
            dropout=self.dropout,
            interactions=self.interactions,
        )
        self.logit_layer = nn.Linear(
            in_features=self.hidden_arl[-1] + self.hidden_aml[-1],
            out_features=1,
        )

    def _generate_layers(self, hidden):
        idx = 1
        while idx < len(hidden):
            yield nn.Linear(hidden[idx-1], hidden[idx])
            yield nn.LayerNorm(hidden[idx])
            yield nn.ReLU()
            yield nn.Dropout(self.dropout)
            idx += 1