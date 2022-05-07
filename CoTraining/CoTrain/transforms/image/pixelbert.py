from .utils import (
    inception_normalize,
    MinMaxResize,
)
from torchvision import transforms
from .randaug import RandAugment
import torchvision.transforms as T


def pixelbert_transform(size=800):
    longer = int((1333 / 800) * size)
    global_transform = transforms.Compose(
        [
            T.Resize(size=256),
            T.RandomResizedCrop(size=(size, size)),
            # MinMaxResize(shorter=size, longer=longer),
            transforms.ToTensor(),
            inception_normalize,
        ]
    )
    local_transform = transforms.Compose(
        [
            T.Resize(size=224),
            T.RandomResizedCrop(size=(96, 96)),
            # MinMaxResize(shorter=size, longer=longer),
            transforms.ToTensor(),
            inception_normalize,
        ]
    )
    return global_transform, local_transform


def pixelbert_transform_randaug(size=800):
    longer = int((1333 / 800) * size)
    global_trs = transforms.Compose(
        [
            T.Resize(size=256),
            T.RandomResizedCrop(size=(size, size)),
            # MinMaxResize(shorter=size, longer=longer),
            transforms.ToTensor(),
            inception_normalize,
        ]
    )
    global_trs.transforms.insert(0, RandAugment(2, 9))
    local_trs = transforms.Compose(
        [
            T.Resize(size=224),
            T.RandomResizedCrop(size=(96, 96)),
            # MinMaxResize(shorter=size, longer=longer),
            transforms.ToTensor(),
            inception_normalize,
        ]
    )
    local_trs.transforms.insert(0, RandAugment(2, 9))
    return global_trs, local_trs
