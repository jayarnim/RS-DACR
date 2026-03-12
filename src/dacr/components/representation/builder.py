from .representation import RepresentationLayer


def rep_fn_builder(**kwargs):
    cls = RepresentationLayer
    return cls(**kwargs)