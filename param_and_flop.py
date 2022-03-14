import copy
import pytorch_lightning as pl
from AllInOne.config import ex
from AllInOne.modules import ViLTransformerSS
from AllInOne.datamodules.multitask_datamodule import MTDataModule
from thop import profile
import torch

@ex.automain
def main(_config):
    _config = copy.deepcopy(_config)
    pl.seed_everything(_config["seed"])

    dm = MTDataModule(_config, dist=True)

    model = ViLTransformerSS(_config)
    input = torch.randn(1, 3, 3, 224, 224)
    macs, params = profile(model, inputs=(input,))
    print(macs, params)

    # 110M