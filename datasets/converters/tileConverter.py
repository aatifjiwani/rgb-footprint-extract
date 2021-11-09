import math
import numpy as np
from osgeo import gdal
import utm

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


def xy2lonlat(x, y, z=22):
    lon = x / pow(2.0, z) * 360.0 - 180
    
    n = math.pi - (2.0 * math.pi * y) / pow(2.0, z)
    lat = math.degrees(math.atan(math.sinh(n)))

    return lon, lat


def array_to_raster(data, 
                    dst_filename,
                    x_min,
                    y_max,
                    wkt_projection,
                    pixel_size=PIXEL_SIZE,
                    data_type=gdal.GDT_Byte):
    """Array > Raster
    Save a raster from a C order array.

    :param array: ndarray
    """
    # You need to get those values like you did.
    rows, cols, bands = data.shape

    driver = gdal.GetDriverByName('GTiff')

    dataset = driver.Create(
        dst_filename,
        cols,
        rows,
        bands,
        data_type)

    dataset.SetGeoTransform((
        x_min,    # 0
        pixel_size,  # 1
        0,                      # 2
        y_max,    # 3
        0,                      # 4
        -pixel_size))  

    dataset.SetProjection(wkt_projection)

    data = np.moveaxis(data, -1, 0)
    for i, image in enumerate(data, 1):
        dataset.GetRasterBand(i).WriteArray(image)
    
    # Write to disk.
    dataset.FlushCache()
    dataset = None


if __name__ == "__main__":
    lon, lat = xy2lonlat(X_MIN, Y_MIN, z=22)
    x, y, _, _ = utm.from_latlon(lat, lon)
    array_to_raster(
        np.zeros((1204, 1024, 3), dtype=np.uint8), 
        "temp.tif", 
        x, 
        y, 
        wkt_projection=WKT_PROJECTION,
        pixel_size=PIXEL_SIZE)