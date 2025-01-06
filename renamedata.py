# in data folder of semantic-segmentation-pytorch, rename segmentations massk in annotations folder to match images in images folder if a mask exists for an image.

def rename():
    import os
    import shutil
    # rename masks to match images
    # get all images
    images = os.listdir('data/images')
    masks = os.listdir('data/annotations')
    for image in images:
        for mask in masks:
            if image == mask:
                continue
            if image.split('.')[0] in mask:
                shutil.move(f'data/annotations/{mask}', f'data/annotations/{image.split(".")[0] + "_mask.png"}')
                print(f'{mask} renamed to {image.split(".")[0] + "_mask.png"}')
    print('done')

rename()