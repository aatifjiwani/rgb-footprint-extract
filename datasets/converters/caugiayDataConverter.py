import numpy as np
import json
import os
from PIL import Image
from glob import glob
from tqdm import tqdm

class CauGiayDataConverter:
    """
        Directory should be in the following structure:

        rootDir/
            gt/
                <LOC>_<ID>.png
            raw/ (filled with raw RBG Images)
                <LOC>_<ID>.png
            images/ (optional)
            masks/ (optional)
    """
    
    def __init__(self, rootDir, target_size=(1024, 1024)):
        self.rootDir = rootDir
        self.gtNames = glob(os.path.join(self.rootDir, "gtl", "*.png"))

        self.target_size = target_size
        
        self.mask_dir = os.path.join(rootDir, "masks")
        if not os.path.isdir(self.mask_dir):
            os.mkdir(self.mask_dir)

        self.img_dir = os.path.join(rootDir, "images")
        if not os.path.isdir(self.img_dir):
            os.mkdir(self.img_dir)
        
    def convertAllToInput(self):
        self.convertToInput(0, len(self.gtNames))
        
    def convertToInput(self, start, end):
        for gtFile in tqdm(self.gtNames):
            basename = os.path.basename(gtFile).split(".")[0]

            # read raw image and groundtruth
            imgFile = os.path.join(self.rootDir, "raw", "%s.png" % basename)
            im = Image.open(imgFile).convert('RGB')
            im = np.array(im)

            gt = Image.open(gtFile).convert('L')
            gt = np.array(gt)

            # tiling and saving
            for y in range(0, im.shape[0], self.target_size[0]):
                for x in range(0, im.shape[1], self.target_size[1]):
                    h, w = self.target_size
                    
                    if y + h > im.shape[0]:
                        h = im.shape[0] - y
                    
                    if x + w > im.shape[1]:
                        w = im.shape[1] - x

                    tile_data = np.zeros(self.target_size + (3,))
                    tile_data[:h, :w, ...] = im[y: y + h, x: x + w, ...]

                    tile_mask = np.zeros(self.target_size)
                    tile_mask[:h, :w] = gt[y: y + h, x: x + w]

                    # np.save(os.path.join(self.mask_dir, "%s_%d_%d_mask.npy" % (basename, y, x)), tile_mask)
                    # np.save(os.path.join(self.img_dir, "%s_%d_%d.npy" % (basename, y, x)), tile_data)

                    tile = Image.fromarray(tile_data.astype(np.uint8))
                    tile.save(os.path.join(self.img_dir, "%s_%d_%d.png" % (basename, y, x)))

                    mask = Image.fromarray(tile_mask.astype(np.uint8))
                    mask.save(os.path.join(self.mask_dir, "%s_%d_%d_mask.png" % (basename, y, x)))

        print("Finished!")


if __name__ == "__main__":
    """
    Example Usage:
        converter = CauGiayDataConverter('/data/CauGiay/')
        converter.convertAllToInput()
    """
    import sys
    converter = CauGiayDataConverter(sys.argv[1])
    converter.convertAllToInput()