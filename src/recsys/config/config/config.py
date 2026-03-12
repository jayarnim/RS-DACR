from dataclasses import dataclass
from typing import Literal, Union
from .pipeline import PipelineCfg
from .trainer import TrainerCfg
from .evaluator import EvaluatorCfg
from .schema import SchemaCfg
from .model import ARLCfg, AMLCfg, ACFCfg


@dataclass
class Config:
    model: Union[ARLCfg, AMLCfg, ACFCfg]
    schema: SchemaCfg
    pipeline: PipelineCfg
    trainer: TrainerCfg
    evaluator: EvaluatorCfg
    strategy: Literal["pointwise", "pairwise", "listwise"]
    model_cls: Literal["arl", "aml", "acf"]
    dataset: str
    seed: int