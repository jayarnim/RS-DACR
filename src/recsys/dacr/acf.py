import torch
import torch.nn as nn
from . import arl, aml
from .components.fusion import FusionLayer
from .components.prediction import ProjectionLayer


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
            arl (nn.Module)
            aml (nn.Moudle)
        """
        super().__init__()

        # attr dictionary for load
        self.init_args = locals().copy()
        del self.init_args["self"]
        del self.init_args["__class__"]

        # global attr
        self.arl = arl
        self.aml = aml
        self.pred_dim = arl.pred_dim + aml.pred_dim

        # generate layers
        self._set_up_components()

    def forward(
        self, 
        user_idx: torch.Tensor, 
        item_idx: torch.Tensor,
    ):
        args = (
            self.arl(user_idx, item_idx),
            self.aml(user_idx, item_idx),
        )
        X_pred = self.fusion(*args)
        return X_pred

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
        X_pred = self.forward(user_idx, item_idx)
        logit = self.prediction(X_pred)
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
        X_pred = self.forward(user_idx, item_idx)
        logit = self.prediction(X_pred)
        return logit

    def _set_up_components(self):
        self._create_components()

    def _create_components(self):
        self.fusion = FusionLayer()

        kwargs = dict(
            dim=self.pred_dim,
        )
        self.prediction = ProjectionLayer(**kwargs)