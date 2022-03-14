# input: (C, T, H, W) output: (C, T, H, W)
def VideoTransform(mode='train', crop_size=224, backend='v100'):
    if backend == 'a100':
        print("initalize data augmentation for a100 gpus")
        import AllInOne.transforms.video_transform as video_transform
        from torchvision import transforms
        # https://github.com/FingerRec/BE/blob/main/src/Contrastive/augment/video_transformations/volume_transforms.py
        if mode == 'train':
            data_transforms = transforms.Compose([
                video_transform.TensorToNumpy(),
                video_transform.Resize(int(crop_size*1.2)),  # 256/224 = 1.14
                video_transform.RandomCrop(crop_size),
                # video_transform.ColorJitter(0.5, 0.5, 0.25, 0.5),  # color operation perimitted, damage attribute
                video_transform.ClipToTensor(channel_nb=3),
                video_transform.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            data_transforms = transforms.Compose([
                video_transform.TensorToNumpy(),
                video_transform.Resize(int(crop_size*1.2)),  # 256
                video_transform.CenterCrop(crop_size),  # 224
                video_transform.ClipToTensor(channel_nb=3),
                video_transform.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        return data_transforms
    else:
        # for pytorch > 1.9.0, V100
        import pytorchvideo.transforms as video_transforms
        # https://pytorchvideo.readthedocs.io/en/latest/api/transforms/transforms.html
        return video_transforms.create_video_transform(mode=mode, min_size=int(crop_size*1.2),
                                                       max_size=int(crop_size*1.5),
                                                       crop_size=crop_size,
                                                       aug_type='randaug',  # randaug/augmix
                                                       num_samples=None)  # not use temporal sub sampling