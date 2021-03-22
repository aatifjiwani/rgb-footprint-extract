from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import os
from tqdm import tqdm

class COCOAnnotationToMask:
    """
        Directory should be in the structure:

        root_dir/
            annotations.json

            /images
                ... list of images from dataset
            /masks (optional)
            
    """
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.coco_client = COCO(os.path.join(self.root_dir, 'annotation.json'))

        self.category_ids = self.coco_client.getCatIds(catNms=['building'])
        self.image_ids = self.coco_client.getImgIds(catIds=self.category_ids)

        self.save_dir = root_dir
        if 'masks' not in os.listdir(root_dir):
            os.mkdir(f"{root_dir}/masks")

    def convert(self):
        no_annotations = 0
        pbar = tqdm(self.image_ids)
        pbar.set_description("{} samples have no annotations".format(no_annotations))

        for img_id in pbar:
            meta = self.coco_client.loadImgs(ids=[img_id])[0]

            # image = io.imread(os.path.join(self.root_dir, "images", meta['file_name']))
            mask = self.generate_mask(meta)
            if mask is None:
                no_annotations += 1
                pbar.set_description("{} samples have no annotations".format(no_annotations))
                continue

            mask = mask.astype(np.uint8)
            np.save(os.path.join(self.save_dir, "masks", meta['file_name'] + "_mask"), mask)    
            
    def generate_mask(self, meta):
        annotation_ids = self.coco_client.getAnnIds(imgIds=meta['id'], catIds=self.category_ids)
        annotations = self.coco_client.loadAnns(annotation_ids)

        mask = np.zeros((meta['height'], meta['width']))
        for ann in annotations:
            mask[:, :] = np.maximum(mask[:, :], self.coco_client.annToMask(ann))

        if len(annotations) == 0:
            return None

        return mask


if __name__ == "__main__":
    """
    Example Usage:

        converter = COCOAnnotationToMask('/data/AICrowd/val')
        converter.convert()
    """
    pass