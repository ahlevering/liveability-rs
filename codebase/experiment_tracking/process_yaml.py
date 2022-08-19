from torchvision import transforms, models

def setup_transforms(transform_cmds):
    compose_list = []
    if 'to_pil' in transform_cmds:
        compose_list.append(transforms.ToPILImage())

    if 'size' in transform_cmds:
        if 'resize' in transform_cmds['size']:
            size = transform_cmds['size']['resize']
            compose_list.append(transforms.Resize([size, size]))
        if 'randomresizedcrop' in transform_cmds['size']:
            compose_list.append(transforms.RandomResizedCrop(
                                                            transform_cmds['size']['randomresizedcrop']['size'],
                                                            transform_cmds['size']['randomresizedcrop']['scale']
                                                        ))
        if 'center_crop' in transform_cmds['size']:
            compose_list.append(transforms.CenterCrop(transform_cmds['size']['crop']))
        if 'five_crop' in transform_cmds['size']:
            compose_list.append(transforms.FiveCrop(transform_cmds['size']['five_crop']))
    
    if 'warps_and_distorts' in transform_cmds:
        if 'hflip' in transform_cmds['warps_and_distorts']:
            compose_list.append(transforms.RandomHorizontalFlip(transform_cmds['warps_and_distorts']['hflip']))
        if 'vflip' in transform_cmds['warps_and_distorts']:
            compose_list.append(transforms.RandomVerticalFlip(transform_cmds['warps_and_distorts']['vflip']))
        if 'randomposterize' in transform_cmds['warps_and_distorts']:
            compose_list.append(transforms.RandomPosterize(
                                transform_cmds['warps_and_distorts']['randomposterize']['bits'],
                                transform_cmds['warps_and_distorts']['randomposterize']['p']))
        if 'jitter' in transform_cmds['warps_and_distorts']:
            compose_list.append(transforms.ColorJitter(
                                brightness=transform_cmds['warps_and_distorts']['jitter']['brightness'],
                                contrast=transform_cmds['warps_and_distorts']['jitter']['contrast'],
                                saturation=transform_cmds['warps_and_distorts']['jitter']['saturation'],
                                hue=transform_cmds['warps_and_distorts']['jitter']['hue']))           
        if 'blur' in transform_cmds['warps_and_distorts']:
            compose_list.append(transforms.GaussianBlur(
                                kernel=transform_cmds['warps_and_distorts']['blur']['kernel'],
                                sigma=transform_cmds['warps_and_distorts']['jitter']['sigma']))
        if 'randomgrayscale' in transform_cmds['warps_and_distorts']:
            compose_list.append(transforms.RandomGrayscale(transform_cmds['warps_and_distorts']['randomgrayscale']))
        if 'rotation' in transform_cmds['warps_and_distorts']:
            compose_list.append(transforms.RandomRotation(transform_cmds['warps_and_distorts']['rotation']))

    if transform_cmds['to_tensor']:
        compose_list.append(transforms.ToTensor())                                
    ## SET TRANSFORMS
    return transforms.Compose(compose_list)