import numpy as np
import georaster
import json
import os
from PIL import Image
from tqdm import tqdm

class Urban3dDataConverter:
    """
        Directory should be in the following structure:

        rootDir/
            gtl/
                <LOC>_Tile_<ID>_GTL.tif
            raw_tif/ (filled with raw RBG Pan Sharpened Tif Images)
                <LOC>_Tile_<ID>_RGB.tif
                <LOC>_Tile_<ID>_DTM.tif
                <LOC>_Tile_<ID>_DSM.tif
            images/ (optional)
            masks/ (optional)
        
    """
    
    def __init__(self, rootDir):
        self.rootDir = rootDir
        self.buildingNames = [file for file in os.listdir(f"{self.rootDir}/gtl/") if '_GTL.tif' in file]
        
        if 'masks' not in os.listdir(rootDir):
            os.mkdir(f"{rootDir}/masks")
            
        if 'images' not in os.listdir(rootDir):
            os.mkdir(f"{rootDir}/images")
        
    def convertAllToInput(self):
        self.convertToInput(0, len(self.buildingNames))
        
    def convertToInput(self, start, end):
        for building in tqdm(self.buildingNames):

            imgFile = building.replace("_GTL", "")
            
            ## Saving mask
            mask = (georaster.SingleBandRaster(f"{self.rootDir}/gtl/{building}").r - 2) / 4
        
            ## Saving input files
            im = Image.open(f"{self.rootDir}/raw_tif/{building.replace('GTL', 'RGB')}") 
            numpyImage = np.array(im) 
            
            numpyDSM = georaster.SingleBandRaster(f"{self.rootDir}/raw_tif/{building.replace('GTL', 'DSM')}")
            numpyDTM = georaster.SingleBandRaster(f"{self.rootDir}/raw_tif/{building.replace('GTL', 'DTM')}")
            numpyNormDSM = (numpyDSM.r - numpyDTM.r)

            mask_dir = "masks" 
            img_dir = "images"

            np.save(f"{self.rootDir}/masks/{imgFile[0:imgFile.index('.')]}_mask", mask)
            np.save(f"{self.rootDir}/images/{imgFile[0:imgFile.index('.')]}", numpyImage)
            np.save(f"{self.rootDir}/{img_dir}/{imgFile[0:imgFile.index('.')]}_NormDSM", numpyNormDSM / 2048)

        print("Finished!")

if __name__ == "__main__":
    """
    Example Usage:
        converter = Urban3dDataConverter('/data/Urban3D/train')
        converter.convertAllToInput()
    """
    pass
                        
            
    
            
            
            
        
        
        
        