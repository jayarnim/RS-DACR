from ..config.model import (
    ARLCfg,
    AMLCfg,
    ACFCfg,
)


def auto(cfg):
    model = cfg["model"]["name"]
    if model=="arl":
        return arl(cfg)
    elif model=="aml":
        return aml(cfg)
    elif model=="acf":
        return acf(cfg)
    else:
        raise ValueError("invalid model name in .yaml config")


def arl(cfg):
    return ARLCfg(
        num_users=cfg["data"]["entity"]["num_users"],
        num_items=cfg["data"]["entity"]["num_items"],
        projection_dim=cfg["model"]["projection_dim"],
        hidden_dim=cfg["model"]["hidden_dim"],
        dropout=cfg["model"]["dropout"],
    )


def aml(cfg):
    return AMLCfg(
        num_users=cfg["data"]["entity"]["num_users"],
        num_items=cfg["data"]["entity"]["num_items"],
        projection_dim=cfg["model"]["projection_dim"],
        hidden_dim=cfg["model"]["hidden_dim"],
        dropout=cfg["model"]["dropout"],
    )


def acf(cfg):
    arl = ARLCfg(
        num_users=cfg["data"]["entity"]["num_users"],
        num_items=cfg["data"]["entity"]["num_items"],
        projection_dim=cfg["model"]["arl"]["projection_dim"],
        hidden_dim=cfg["model"]["arl"]["hidden_dim"],
        dropout=cfg["model"]["arl"]["dropout"],
    )
    aml = AMLCfg(
        num_users=cfg["data"]["entity"]["num_users"],
        num_items=cfg["data"]["entity"]["num_items"],
        projection_dim=cfg["model"]["aml"]["projection_dim"],
        hidden_dim=cfg["model"]["aml"]["hidden_dim"],
        dropout=cfg["model"]["aml"]["dropout"],
    )
    return ACFCfg(
        arl=arl,
        aml=aml,
    )