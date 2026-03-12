import torch
import torch.nn as nn
from . import arl, aml
from .components.scorer import LinearProjectionLayer


class Module(nn.Module):
    def __init__(
        self,
        arl: nn.Module,
        aml: nn.Module,
    ):
        """
        Deep collaborative recommendation algorithm based on attention mechanism (Cui et al., 2022)
        -----
        Implements the base structure of Deep Collaborative Recommendation Algorithm Based on Attention Mechanism (DACR),
        MF, MLP & history embedding based latent factor model,
        applying attention mechanism to denoise from implicit feedback,
        combining a Attention Representation Learning (aRL) and a Attention Matching Function Learning Networks (aML)
        to learn low-rank linear represenation & high-rank nonlinear user-item interactions.

        Args:
            n_users (int):
                total number of users in the dataset, U.
            n_items (int):
                total number of items in the dataset, I.
            n_factors (int):
                dimensionality of user and item latent representation vectors, K.
            hidden_rl (list):
                layer dimensions for the representation @ RLNet. 
                (e.g., [64, 32, 16, 8])
            hidden_ml (list): 
                layer dimensions for the matching function @ MLNet. 
                (e.g., [64, 32, 16, 8])
            dropout (float): 
                dropout rate applied to MLP layers for regularization.
            interaction (torch.Tensor): 
                user-item interaction matrix, masked evaluation datasets. 
                (shape: [U+1, I+1])
        """
        super().__init__()

        # attr dictionary for load
        self.init_args = locals().copy()
        del self.init_args["self"]
        del self.init_args["__class__"]

        # global attr
        self.arl = arl
        self.aml = aml
        self.predictive_dim = arl.predictive_dim + aml.predictive_dim

        # generate layers
        self._set_up_components()

    def forward(
        self, 
        user_idx: torch.Tensor, 
        item_idx: torch.Tensor,
    ):
        # modules
        PREDICTIVE_VECTORS = (
            self.arl(user_idx, item_idx),
            self.aml(user_idx, item_idx),
        )

        # fusion
        kwargs = dict(
            tensors=PREDICTIVE_VECTORS, 
            dim=-1,
        )
        predictive_vec = torch.cat(**kwargs)

        return predictive_vec

    def estimate(
        self, 
        user_idx: torch.Tensor, 
        item_idx: torch.Tensor,
    ):
        """
        Training Method
        -----

        Args:
            user_idx (torch.Tensor): target user idx (shape: [B,])
            item_idx (torch.Tensor): target item idx (shape: [B,])
        
        Returns:
            logit (torch.Tensor): (u,i) pair interaction logit (shape: [B,])
        """
        predictive_vec = self.forward(user_idx, item_idx)
        logit = self.scorer(predictive_vec)
        return logit

    @torch.no_grad()
    def predict(
        self, 
        user_idx: torch.Tensor, 
        item_idx: torch.Tensor,
    ):
        """
        Evaluation Method
        -----

        Args:
            user_idx (torch.Tensor): target user idx (shape: [B,])
            item_idx (torch.Tensor): target item idx (shape: [B,])

        Returns:
            prob (torch.Tensor): (u,i) pair interaction probability (shape: [B,])
        """
        predictive_vec = self.forward(user_idx, item_idx)
        logit = self.scorer(predictive_vec)
        return logit

    def _set_up_components(self):
        self._create_components()

    def _create_components(self):
        kwargs = dict(
            input_dim=self.predictive_dim,
        )
        self.scorer = LinearProjectionLayer(**kwargs)