from ..config.model import ARLCfg, AMLCfg, ACFCfg


def model(cfg):
    cls = cfg["model"]["name"]

    if cls=="arl":
        return arl(cfg)
    elif cls=="aml":
        return aml(cfg)
    elif cls=="acf":
        return acf(cfg)
    else:
        raise ValueError("invalid model name in .yaml config")


def arl(cfg):
    return ARLCfg(
        num_users=cfg["data"]["entity"]["num_users"],
        num_items=cfg["data"]["entity"]["num_items"],
        params=cfg["model"]["params"],
    )


def aml(cfg):
    return AMLCfg(
        num_users=cfg["data"]["entity"]["num_users"],
        num_items=cfg["data"]["entity"]["num_items"],
        params=cfg["model"]["params"],
    )


def acf(cfg):
    arl = ARLCfg(
        num_users=cfg["data"]["entity"]["num_users"],
        num_items=cfg["data"]["entity"]["num_items"],
        params=cfg["model"]["params"]["arl"],
    )
    aml = AMLCfg(
        num_users=cfg["data"]["entity"]["num_users"],
        num_items=cfg["data"]["entity"]["num_items"],
        params=cfg["model"]["params"]["aml"],
    )
    return ACFCfg(
        arl=arl,
        aml=aml,
    )