import os
import argparse
from PIL import Image
import numpy as np
import maxflow
import utm

from utils.geo import xy2lonlat, array_to_raster

SMOOTHING = 110
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
    parser.add_argument('--tile-size-x', type=int, default=1024, 
                        help='tile size')
    parser.add_argument('--tile-size-y', type=int, default=1024, 
                        help='tile size')
    parser.add_argument('--target-size-x', type=int, default=20480, 
                        help='tile size')
    parser.add_argument('--target-size-y', type=int, default=20480, 
                        help='tile size')
    parser.add_argument('--zoom-level', type=int, default=22, 
                        help='original Google Earth image zoom level.')
    parser.add_argument('--pixel-size', type=float, default=0.037, 
                        help='original Google Earth image pixel size in meters.')
    args = parser.parse_args()
    return args


def test():
    args = _parse_args()
    from glob import glob
    for tile_path in glob(os.path.join(args.input_dir, "*.png")):
        basename = os.path.basename(tile_path).split(".")[0]
        
        tile = Image.open(tile_path).convert("RGB")
        tile = np.array(tile)

        _, y, x = basename.split("_")
        
        lon, lat = xy2lonlat(X_MIN + int(x) // 256, 
                             Y_MIN + int(y) // 256, 
                             z=args.zoom_level)

        utm_east, utm_north, _, _ = utm.from_latlon(lat, lon)

        array_to_raster(
            tile, 
            os.path.join(args.output_dir, "%d_%d.tif" % (Y_MIN + int(y) // 256, X_MIN + int(x) // 256)), 
            utm_east, 
            utm_north, 
            wkt_projection=WKT_PROJECTION,
            pixel_size=args.pixel_size)


def graph_denoise(img, smoothing=110):
    # Create the graph.
    g = maxflow.Graph[int]()
    # Add the nodes. nodeids has the identifiers of the nodes in the grid.
    nodeids = g.add_grid_nodes(img.shape)
    # Add non-terminal edges with the same capacity.
    g.add_grid_edges(nodeids, smoothing)
    # Add the terminal edges. The image pixels are the capacities
    # of the edges from the source node. The inverted image pixels
    # are the capacities of the edges to the sink node.
    g.add_grid_tedges(nodeids, img, 255 - img)
    # Find the maximum flow.
    g.maxflow()
    # Get the segments of the nodes in the grid.
    sgm = g.get_grid_segments(nodeids)
    # The labels should be 1 where sgm is False and 0 otherwise.
    img_denoised = np.logical_not(sgm).astype(np.uint8) * 255

    return img_denoised


def main():
    args = _parse_args()

    assert args.target_size_x % args.tile_size_x == 0
    assert args.target_size_y % args.tile_size_y == 0

    for y in range(0, 131328, args.target_size_y):
        for x in range(0, 96256, args.target_size_x):
            if y + args.target_size_y > 131328:
                y = 131328 - args.target_size_y
        
            if x + args.target_size_x > 96256:
                x = 96256 - args.target_size_x

            # constructing merged-image
            im = np.zeros((args.target_size_y, args.target_size_x, 1), dtype=np.uint8)
            for tile_y in range(0, args.target_size_y, args.tile_size_y):
                for tile_x in range(0, args.target_size_x, args.tile_size_x):
                    # read tile data
                    tile_path = os.path.join(args.input_dir, "q1_%d_%d.png" % (tile_y + y, tile_x + x))
                    tile = Image.open(tile_path).convert("L")
                    tile = np.array(tile)
                    # remove un-connected noise
                    tile = graph_denoise(tile)

                    im[tile_y: tile_y + args.tile_size_y, tile_x: tile_x + args.tile_size_x, 0] = tile
            
            # push image to Geo Tiff
            lon, lat = xy2lonlat(X_MIN + int(x) // 256, 
                                 Y_MIN + int(y) // 256, 
                                 z=args.zoom_level)

            utm_east, utm_north, _, _ = utm.from_latlon(lat, lon)

            array_to_raster(
                im, 
                os.path.join(args.output_dir, "%d_%d.tif" % (Y_MIN + int(y) // 256, X_MIN + int(x) // 256)), 
                utm_east, 
                utm_north, 
                wkt_projection=WKT_PROJECTION,
                pixel_size=args.pixel_size,
                no_data_value=0)


if __name__ == "__main__":
    main()
    # test()