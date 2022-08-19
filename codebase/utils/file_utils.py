from pathlib import Path

def recursively_get_files(base_dir, extensions=['.tif', '.jpg', '.png', '.tiff']):
    file_list = []
    for filetype in extensions:
        file_list.extend([str(img_path) for img_path in Path(base_dir).glob('**/*{}'.format(filetype))])
    return file_list

def get_image_label_pairs(images_root, labels_root):
    images = recursively_get_files(images_root)
    labels = recursively_get_files(labels_root, ['.geojson'])

    img_label_pairs = []
    for img in images:
        # Remove extension
        img_city = Path(img).stem.split('_')[0]
        label_file = [l for l in labels if img_city in Path(l).stem.split('_')[0]][0]
        img_label_pairs.append([img, label_file])
    return img_label_pairs