from dataclasses import dataclass


@dataclass
class ARLCfg:
    num_users: int
    num_items: int
    projection_dim: int
    hidden_dim: list
    dropout: float


@dataclass
class AMLCfg:
    num_users: int
    num_items: int
    projection_dim: int
    hidden_dim: list
    dropout: float


@dataclass
class ACFCfg:
    arl: ARLCfg
    aml: AMLCfg