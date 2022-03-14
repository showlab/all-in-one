# pretrain dataset
## video
from .webvid_datamodule import WEBVIDDataModule
from .howto100m_datamodule import HT100MDataModule
from .yttemporal_datamodule import YTTemporalMDataModule
## image
from .cc3m_datamodule import CC3MDataModule
from .vg_caption_datamodule import VisualGenomeCaptionDataModule
from .coco_caption_karpathy_datamodule import CocoCaptionKarpathyDataModule
from .conceptual_caption_datamodule import ConceptualCaptionDataModule
from .sbu_datamodule import SBUCaptionDataModule
# finetune dataset
## image
from .f30k_caption_karpathy_datamodule import F30KCaptionKarpathyDataModule
from .vqav2_datamodule import VQAv2DataModule
from .nlvr2_datamodule import NLVR2DataModule
from .msrvtt_datamodule import MSRVTTDataModule
from .msrvttqa_datamodule import MSRVTTQADataModule
from .msrvtt_choice_datamodule import MSRVTTChoiceDataModule
from .msvd_datamodule import MSVDDataModule
from .msvdqa_datamodule import MSVDQADataModule
from .vcr_datamodule import VCRDataModule
## video
from .ego4d_datamodule import Ego4DDataModule
from .tvqa_datamodule import TVQADataModule
from .lsmdc_choice_datamodule import LSMDCChoiceDataModule
from .ego4d_choice_datamodule import EGO4DChoiceDataModule
from .tgif_datamodule import TGIFDataModule
from .tgifqa_datamodule import TGIFQADataModule
from .didemo_datamodule import DIDEMODataModule
from .hmdb51_datamodule import HMDB51DataModule
from .k400_datamodule import K400DataModule
from .lsmdc_datamodule import LSMDCDataModule
from .activitynet_datamodule import ActivityNetDataModule

_datamodules = {
    "vg": VisualGenomeCaptionDataModule,
    "f30k": F30KCaptionKarpathyDataModule,
    "coco": CocoCaptionKarpathyDataModule,
    "gcc": ConceptualCaptionDataModule,
    "sbu": SBUCaptionDataModule,
    "vqa": VQAv2DataModule,
    "nlvr2": NLVR2DataModule,
    "cc3m": CC3MDataModule,
    'howto100m': HT100MDataModule,
    'webvid': WEBVIDDataModule,
    'msrvtt': MSRVTTDataModule,
    'msrvttqa': MSRVTTQADataModule,
    'msrvtt_choice': MSRVTTChoiceDataModule,
    'msvd': MSVDDataModule,
    'msvdqa': MSVDQADataModule,
    'vcr': VCRDataModule,
    'ego4d': Ego4DDataModule,
    'tvqa': TVQADataModule,
    'lsmdc_choice': LSMDCChoiceDataModule,
    'ego4d_choice': EGO4DChoiceDataModule,
    'yttemporal': YTTemporalMDataModule,
    'tgif': TGIFDataModule,
    "tgifqa": TGIFQADataModule,
    'didemo': DIDEMODataModule,
    'hmdb51': HMDB51DataModule,
    'k400': K400DataModule,
    'lsmdc': LSMDCDataModule,
    'activitynet': ActivityNetDataModule
}
