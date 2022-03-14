from AllInOne.datasets import MSRVTTChoiceDataset
from .datamodule_base import BaseDataModule


class MSRVTTChoiceDataModule(BaseDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def dataset_cls(self):
        return MSRVTTChoiceDataset

    @property
    def dataset_cls_no_false(self):
        return MSRVTTChoiceDataset

    @property
    def dataset_name(self):
        return "msrvtt_choice"
