from dataclasses import dataclass


@dataclass
class ARLCfg:
    num_users: int
    num_items: int
    params: dict


@dataclass
class AMLCfg:
    num_users: int
    num_items: int
    params: dict


@dataclass
class ACFCfg:
    arl: ARLCfg
    aml: AMLCfg