from ssd.modeling.anchors.prior_box import PriorBox
from .target_transform import SSDTargetTransform
from .transforms import *


def build_transforms(cfg, is_train=True):
    if is_train:
        transform = [
            ConvertFromInts(), # Convert img to float32
            PhotometricDistort(), # Random HSV on image
            #Expand(cfg.INPUT.PIXEL_MEAN), # increase image size padding with mean value
            RandomSampleCrop(), # Crop image
            RandomMirror(), # Flip right-left randomly
            ToPercentCoords(), # Normalize BBox Cords
            Resize(cfg.INPUT.IMAGE_SIZE), # Resize input image (don't preserve aspect ration)
            SubtractMeans(cfg.INPUT.PIXEL_MEAN), # subtract mean from image pixels
            ToTensor(),
        ]
    else:
        transform = [
            ConvertFromInts(), # Convert img to float32
            ToPercentCoords(), # Normalize BBox Cords
            Resize(cfg.INPUT.IMAGE_SIZE),
            SubtractMeans(cfg.INPUT.PIXEL_MEAN),
            ToTensor()
        ]
    transform = Compose(transform)
    return transform


def build_target_transform(cfg):
    transform = SSDTargetTransform(PriorBox(cfg)(),
                                   cfg.MODEL.CENTER_VARIANCE,
                                   cfg.MODEL.SIZE_VARIANCE,
                                   cfg.MODEL.THRESHOLD)
    return transform
