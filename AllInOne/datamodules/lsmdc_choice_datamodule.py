from AllInOne.datasets import LSMDCChoiceDataset
from .datamodule_base import BaseDataModule


class LSMDCChoiceDataModule(BaseDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def dataset_cls(self):
        return LSMDCChoiceDataset

    @property
    def dataset_cls_no_false(self):
        return LSMDCChoiceDataset

    @property
    def dataset_name(self):
        return "lsmdc_choice"
