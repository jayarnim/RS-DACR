import torch
import torch.nn as nn
from .components.embedding import HistoryEmbedding
from .components.representation.builder import rep_fn_builder
from .components.matching.builder import matching_fn_builder
from .components.scorer import LinearProjectionLayer


class Module(nn.Module):
    def __init__(
        self,
        interactions: torch.Tensor, 
        num_users: int,
        num_items: int,
        projection_dim: int,
        hidden_dim: list,
        dropout: float,
    ):
        """
        Deep collaborative recommendation algorithm based on attention mechanism (Cui et al., 2022)
        -----
        Implements the base structure of Attention Representation Learning (aRL),
        MF & history embedding based latent factor model,
        applying attention mechanism to denoise from implicit feedback,
        sub-module of Deep Collaborative Recommendation Algorithm Based on Attention Mechanism (DACR)
        to learn low-rank linear represenation.

        Args:
            interactions (torch.Tensor): 
                user-item interaction matrix, masked evaluation datasets. 
                (shape: [U+1, I+1])
            num_users (int): 
                total number of users in the dataset, U.
            num_items (int): 
                total number of items in the dataset, I.
            projection_dim (int): 
                dimensionality of user and item projection vectors.
            hidden_dim (list): 
                layer dimensions for the representation. 
                (e.g., [128, 64, 32])
            dropout (float): 
                dropout rate applied to MLP layers for regularization.
        """
        super().__init__()

        # attr dictionary for load
        self.init_args = locals().copy()
        del self.init_args["self"]
        del self.init_args["__class__"]

        # global attr
        self.interactions = interactions
        self.num_users = num_users
        self.num_items = num_items
        self.projection_dim = projection_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.predictive_dim = hidden_dim[-1]

        # generate layers
        self._set_up_components()

    def forward(
        self, 
        user_idx: torch.Tensor, 
        item_idx: torch.Tensor,
    ):
        user_emb, item_emb = self.embedding(user_idx, item_idx)
        user_rep, item_rep = self.representation(user_emb, item_emb)
        predictive_vec = self.matching(user_rep, item_rep)
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
            logit (torch.Tensor): (u,i) pair interaction logit (shape: [B,])
        """
        predictive_vec = self.forward(user_idx, item_idx)
        logit = self.scorer(predictive_vec)
        return logit

    def _set_up_components(self):
        self._create_components()

    def _create_components(self):
        kwargs = dict(
            interactions=self.interactions, 
            num_users=self.num_users,
            num_items=self.num_items,
            projection_dim=self.projection_dim,
        )
        self.embedding = HistoryEmbedding(**kwargs)

        kwargs = dict(
            input_dim=self.projection_dim*2,
            hidden_dim=self.hidden_dim,
            dropout=self.dropout,
        )
        self.representation = rep_fn_builder(**kwargs)

        kwargs = dict(
            name="mf",
        )
        self.matching = matching_fn_builder(**kwargs)

        kwargs = dict(
            input_dim=self.hidden_dim[-1],
        )
        self.scorer = LinearProjectionLayer(**kwargs)