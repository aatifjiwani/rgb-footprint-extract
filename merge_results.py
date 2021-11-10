import os
import argparse

import numpy as np
import utm

from utils.geo import xy2lonlat, array_to_raster

X_MIN, Y_MIN = 3340084, 1970605
PIXEL_SIZE = 0.037 # in m, level 22
WKT_PROJECTION = """
        PROJCS["WGS 84 / UTM zone 48N",
            GEOGCS["WGS 84",
                DATUM["WGS_1984",
                    SPHEROID["WGS 84",6378137,298.257223563,
                        AUTHORITY["EPSG","7030"]],
                    AUTHORITY["EPSG","6326"]],
                PRIMEM["Greenwich",0,
                    AUTHORITY["EPSG","8901"]],
                UNIT["degree",0.0174532925199433,
                    AUTHORITY["EPSG","9122"]],
                AUTHORITY["EPSG","4326"]],
            PROJECTION["Transverse_Mercator"],
            PARAMETER["latitude_of_origin",0],
            PARAMETER["central_meridian",105],
            PARAMETER["scale_factor",0.9996],
            PARAMETER["false_easting",500000],
            PARAMETER["false_northing",0],
            UNIT["metre",1,
                AUTHORITY["EPSG","9001"]],
            AXIS["Easting",EAST],
            AXIS["Northing",NORTH],
            AUTHORITY["EPSG","32648"]]
    """


def _parse_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--input-dir', type=str, 
                        help='input dir contains tiles predictions')
    parser.add_argument('--output-dir', type=str, 
                        help='output dir contains merged results')
    args = parser.parse_args()
    return args

def main():
    args = _parse_args()

    for y in range(0, 131950, 6500):
        for x in range(0, 96850, 6500):

if __name__ == "__main__":
    main()