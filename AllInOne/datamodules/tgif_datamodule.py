from AllInOne.datasets import TGIFDataset
from .datamodule_base import BaseDataModule


class TGIFDataModule(BaseDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def dataset_cls(self):
        return TGIFDataset

    @property
    def dataset_cls_no_false(self):
        return TGIFDataset

    @property
    def dataset_name(self):
        return "tgif"
