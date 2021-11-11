import os
import argparse

import numpy as np
import utm

from utils.geo import xy2lonlat, array_to_raster

X_MIN, Y_MIN = 3340084, 1970605
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
    parser.add_argument('--tile-size-x', type=int, default=650, 
                        help='tile size')
    parser.add_argument('--tile-size-y', type=int, default=650, 
                        help='tile size')
    parser.add_argument('--target-size-x', type=int, default=6500, 
                        help='tile size')
    parser.add_argument('--target-size-y', type=int, default=6500, 
                        help='tile size')
    parser.add_argument('--zoom-level', type=int, default=22, 
                        help='original Google Earth image zoom level.')
    parser.add_argument('--pixel-size', type=float, default=0.037, 
                        help='original Google Earth image pixel size in meters.')
    args = parser.parse_args()
    return args


def main():
    args = _parse_args()

    assert args.target_size_x % args.tile_size_x == 0
    assert args.target_size_y % args.tile_size_y == 0

    for y in range(0, 131950, args.target_size_y):
        for x in range(0, 96850, args.target_size_x):
            im = np.zeros((args.target_size_y, args.target_size_x), dtype=np.uint8)
            for tile_y in range(0, args.target_size_y, args.tile_size_y):
                for tile_x in range(0, args.target_size_x, args.tile_size_x):
                    tile = np.load("q1_%d_%d.npz" % (tile_y + y, tile_x + x))
                    tile = tile["arr_0"]

                    im[tile_y: tile_y + args.tile_size_y, tile_x: tile_x + args.tile_size_x] = tile
            
            lon, lat = xy2lonlat(X_MIN + x, Y_MIN + y, z=args.zoom_level)
            utm_east, utm_north, _, _ = utm.from_latlon(lat, lon)
            array_to_raster(
                im, 
                os.path.join(args.output_dir, "%d_%d.tif" % (y, x)), 
                utm_east, 
                utm_north, 
                wkt_projection=WKT_PROJECTION,
                pixel_size=args.pixel_size)


if __name__ == "__main__":
    main()