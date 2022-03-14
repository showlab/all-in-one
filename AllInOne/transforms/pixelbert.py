from .utils import (
    inception_normalize,
    MinMaxResize,
)
from torchvision import transforms
from .randaug import RandAugment
import torchvision.transforms as T

def pixelbert_transform(size=800):
    longer = int((1333 / 800) * size)
    return transforms.Compose(
        [
            T.Resize(size=256),
            T.RandomResizedCrop(size=(size, size)),
            # MinMaxResize(shorter=size, longer=longer),
            transforms.ToTensor(),
            inception_normalize,
        ]
    )


def pixelbert_transform_randaug(size=800):
    longer = int((1333 / 800) * size)
    trs = transforms.Compose(
        [
            T.Resize(size=256),
            T.RandomResizedCrop(size=(size, size)),
            # MinMaxResize(shorter=size, longer=longer),
            transforms.ToTensor(),
            inception_normalize,
        ]
    )
    trs.transforms.insert(0, RandAugment(2, 9))
    return trs
