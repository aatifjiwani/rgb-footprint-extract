import geopandas as gpd
import numpy as np
import os
import pyproj
from pyproj import Geod
import rasterio
import rasterio.features
from skimage.draw import  polygon2mask
import shapely
from shapely.geometry import box
from shapely.ops import transform, orient


# Params
OAK_FP = '/oak/stanford/groups/deho/building_compliance/'
TIF_FP = '/oak/stanford/groups/deho/building_compliance/san_jose_naip_512/raw_tif'


def get_footprint_gpd(mask, file_name, size):
    """
    Converts predictions from np.array to shapely polygons
    :param mask: (np.array) Footprint mask from which to generate the polygons
    :param file_name: (str) Name of the TIF file 
    :param size: (int) Resolution size [512, 1024]
    :return: (gpd.GeoDataFrame)
    """

    with rasterio.open(os.path.join(TIF_FP, f'{file_name}.tif')) as ds:
        t = ds.transform

    # Adjust for resolution
    if size == 512:
        factor = 1
    elif size == 1024:
        factor = 2
    else:
        raise NotImplemented('[ERROR] GPD Footprint --- Check resolution')

    shapes = rasterio.features.shapes(mask, connectivity=4)
    polygons = [shapely.geometry.Polygon(shape[0]["coordinates"][0]) for shape in shapes if shape[1] == 1]
    polygons = [shapely.affinity.affine_transform(geom, [t.a / factor, t.b, t.d, t.e / factor, t.xoff, t.yoff]) for geom
                in polygons]
    buildings = gpd.GeoDataFrame(geometry=polygons, crs='EPSG:26910')
    buildings = buildings.to_crs('EPSG:4326')
    return buildings


# Pad small gt buildings
def pad_small_buildings(buildings_gpd, pad_buffer):
    """
    Creates a buffer/padding of size pad_buffer around the footprints 
    in buildings_gpd
    :param buildings_gpd: (gpd.GeoDataFrame) of footprints
    :param pad_buffer: (int)
    :return: (gpd.GeoDataFrame)
    """
    buildings_gpd = buildings_gpd.to_crs(crs=3857)
    buildings_gpd.geometry = buildings_gpd.geometry.buffer(pad_buffer)
    buildings_gpd = buildings_gpd.to_crs('EPSG:4326')
    return buildings_gpd


def separate_buildings(buildings_gpd, buffer_val):
    """
    Separates loosely connected building footprints in buildings_gpd
    :param buildings_gpd: (gpd.GeoDataFrame) of footprints
    :param buffer_val: (int)
    :return: (gpd.GeoDataFrame)
    """
    buildings_gpd = buildings_gpd.to_crs(crs=3857)
    buildings_gpd.geometry = buildings_gpd.geometry.buffer(-buffer_val)
    buildings_gpd = buildings_gpd.explode(index_parts=False)
    buildings_gpd.geometry = buildings_gpd.geometry.buffer(buffer_val)
    buildings_gpd = buildings_gpd.to_crs('EPSG:4326')
    return buildings_gpd


def filter_roads(buildings_gpd, file_name, road_buffer):
    """
    Removes footprints in buildings_gpd that lie on roads
    :param buildings_gpd: (gpd.GeoDataFrame)
    :param file_name: (str)
    :param road_buffer: (int)
    :return: (gpd.GeoDataFrame)
    """

    # Get zoning data
    zoning = gpd.read_file(os.path.join(OAK_FP, 'san_jose_suppl', 'san_jose_Zoning_Districts.geojson'))
    zoning = zoning[(zoning['ZONINGABBREV'].str.contains('R-')) | \
                    ((zoning['ZONINGABBREV'] == 'A(PD)') & (zoning['PDUSE'] == 'Res'))]

    # Extract bounds from TIF
    with rasterio.open(os.path.join(TIF_FP, f'{file_name}.tif')) as inds:
        bounds = inds.bounds
        geom = box(*bounds)

    # Convert TIF bounds to standard 4326
    wgs84 = pyproj.CRS('EPSG:26910')  # LA is 11, SJ is 10
    utm = pyproj.CRS('EPSG:4326')
    project = pyproj.Transformer.from_crs(wgs84, utm, always_xy=True).transform
    utm_geom = shapely.ops.transform(project, geom)

    # Clip the residential zones to just the TIF bounds
    zoning_clipped = gpd.clip(zoning, utm_geom)

    ## Filter predictions on roads (and other non-residential areas)
    zoning_clipped = zoning_clipped.to_crs(crs=3857)
    zoning_clipped = zoning_clipped.buffer(road_buffer)
    zoning_clipped = zoning_clipped.to_crs('EPSG:4326')

    buildings_gpd = gpd.clip(buildings_gpd, zoning_clipped)
    return buildings_gpd


def filter_buildings_area(buildings_gpd, area_thresh, larger_than=True):
    """
    Filters buildings_gpd for buildings that meet a specified minimum or maximum
    area threshold
    :param buildings_gpd: (gpd.GeoDataFrame)
    :param area_thresh: (int) square meters
    :param larger_than: (bool) True if filtering for buildings with area
    larger than a threshold
    :return: (gpd.GeoDataFrame)
    """

    geod = Geod(ellps="WGS84")

    # apply orient() before passing to Geod so that the area is not negative
    buildings_gpd['area'] = buildings_gpd['geometry'].progress_apply(
        lambda x: geod.geometry_area_perimeter(orient(x))[0])

    if larger_than:
        buildings_gpd = buildings_gpd.loc[buildings_gpd['area'] > area_thresh]
    else:
        buildings_gpd = buildings_gpd.loc[buildings_gpd['area'] < area_thresh]
    buildings_gpd = buildings_gpd.to_crs('EPSG:4326')
    return buildings_gpd


def get_nparray_from_gpd(building_gpd, file_name, size):
    """
    Generates a mask (np.array) of building footprints from the polygons in building_gpd
    :param building_gpd: (gpd.GeoDataFrame)
    :param file_name: (str)
    :param size: (int)
    :return: (np.array)
    """
    # Create empty array
    building_np = np.zeros((size, size), dtype="int64")

    # Get spatial info from TIF
    with rasterio.open(os.path.join(TIF_FP, f'{file_name}.tif')) as ds:
        t = ds.transform

    # Transform polygons to pixel space
    building_gpd = building_gpd.to_crs(crs='EPSG:26910')
    building_gpd = building_gpd.explode(column='geometry', index_parts=False, ignore_index=True)

    polygons = building_gpd.geometry

    # Adjust for resolution
    if size == 512:
        factor = 1
    elif size == 1024:
        factor = 2
    else:
        raise NotImplemented('[ERROR] GPD Footprint --- Check resolution')
    A = t.a / factor
    E = t.e / factor

    # Get inverse transformation
    k = 1 / (1 - (t.b * t.d) / (A * E))
    polygons = [shapely.affinity.affine_transform(
        geom,
        [k / A,  # a
         -(t.b * k) / (A * E),  # b
         -(t.d * k) / (E * A),  # d
         1 / E + (t.d * t.b * k) / (E * A * E),  # e
         (k / A) * ((t.b * t.yoff) / (E) - t.xoff),  # xOff
         -(t.d * k * t.b * t.yoff) / (E * A * E) + (t.d * k * t.xoff) / (E * A) - t.yoff / E]) for geom in
        polygons]  # yOff

    # Create mask for each polygon
    for poly in polygons:
        a = poly.exterior.coords.xy
        poly_mask = polygon2mask((size, size), polygon=list(zip(a[1], a[0])))
        building_np += poly_mask

    # Flatten
    building_np = (building_np > 0).astype(np.int32)

    return building_np


def get_pred_small_buildings(pred_image, file_name, buffer_val, small_area_thresh,
                             large_area_thresh, road_buffer, return_np=True):
    """
    Retrieves a np.array or gpd.GeoDataFrame of the small buildings in a prediction
    :param pred_image: (np.array) prediction
    :param file_name: (str) TIF file name
    :param buffer_val: (int) Buffer for building separation
    :param small_area_thresh: (int) Minimum small building size
    :param large_area_thresh: (int) Maximum small building size
    :param road_buffer: (int) Buffer for road prediction removal
    :param return_np: (bool) True to return a np.array, else a gpd.GeoDataFrame
    :return: (np.array) or (gpd.GeoDataFrame) depending on return_np param
    """

    # Get gpd of building footprints
    inferred_buildings = get_footprint_gpd(mask=pred_image, file_name=file_name, size=pred_image.shape[0])

    # Separate closely connected buildings
    inferred_buildings = separate_buildings(buildings_gpd=inferred_buildings, buffer_val=buffer_val)

    # Filter out predictions on roads
    inferred_buildings = filter_roads(buildings_gpd=inferred_buildings, file_name=file_name, road_buffer=road_buffer)

    # Filter out predictions that are too small
    inferred_buildings = filter_buildings_area(
        buildings_gpd=inferred_buildings, area_thresh=small_area_thresh, larger_than=True)

    # Filter for small buildings
    inferred_buildings = filter_buildings_area(
        buildings_gpd=inferred_buildings, area_thresh=large_area_thresh, larger_than=False)

    if not return_np:
        return inferred_buildings

    # Convert from GPD to np.array
    pred_small_build = get_nparray_from_gpd(building_gpd=inferred_buildings, file_name=file_name,
                                            size=pred_image.shape[0])

    return pred_small_build


def generate_metric_mask(
        bg_gt, bg_pred, build_gt, build_pred, file_name,
        pad_buffers, buffer_val, small_area_thresh, large_area_thresh,
        road_buffer):
    """
    Generates the gt and predicted background and foreground masks to be used
    in the IoU computation for each class.
    :param bg_gt: (np.array) type np.int32
    :param bg_pred: (np.array) type np.int32
    :param build_gt: (np.array) type np.int32
    :param build_pred: (np.array) type np.int32
    :param file_name: (str)
    :param pad_buffers: (list of int) small building padding buffers
    :param buffer_val: (int) buffer for small building separation
    :param small_area_thresh: (int) Minimum area for small buildings
    :param large_area_thresh: (int) Maximum area for small buildings
    :param road_buffer: (int) Buffer to filter road predictions
    :return: (tuple) of dict with keys [SmIoU-V1, SmIoU-V2, mIoU-SB]
    """

    # Generate IoU masks for each image in batch
    gt_mask_bg, pred_mask_bg = {'mIoU-SB': np.zeros_like(build_gt)}, {'mIoU-SB': np.zeros_like(build_gt)}
    for b in pad_buffers:
        # Only save one for background as both SmIoU versions share the same masks
        gt_mask_bg['SmIoU-{}'.format(b)], pred_mask_bg['SmIoU-{}'.format(b)] = np.zeros_like(build_gt), np.zeros_like(
            build_gt)

    gt_mask_bd = {'mIoU-SB': np.zeros_like(build_gt), 'SmIoU-V1': np.zeros_like(build_gt), 'SmIoU-V2': np.zeros_like(build_gt)}
    pred_mask_bd = {'mIoU-SB': np.zeros_like(build_gt), 'SmIoU-V1': np.zeros_like(build_gt), 'SmIoU-V2': np.zeros_like(build_gt)}

    for i in range(gt_mask_bg['mIoU-SB'].shape[0]):
        # Class 0 (bg) ---------------------
        # * Get gt small buildings
        build_gt_gpd = get_footprint_gpd(
            mask=build_gt[i, :], file_name=file_name[i], size=build_gt.shape[1])
        small_build_gt_gpd = filter_buildings_area(
            buildings_gpd=build_gt_gpd, area_thresh=small_area_thresh, larger_than=True)
        small_build_gt_gpd = filter_buildings_area(
            buildings_gpd=small_build_gt_gpd, area_thresh=large_area_thresh, larger_than=False)
        small_build_gt_np = get_nparray_from_gpd(
            small_build_gt_gpd, file_name[i], build_pred.shape[1])

        # * Get predicted small buildings
        small_build_pred_np = get_pred_small_buildings(
            pred_image=build_pred[i, :], file_name=file_name[i], buffer_val=buffer_val,
            small_area_thresh=small_area_thresh, large_area_thresh=large_area_thresh,
            road_buffer=road_buffer)

        # mIoU (Small buildings)
        gt_mask_bg['mIoU-SB'][i] = (1 - small_build_gt_np).astype(np.int32)
        pred_mask_bg['mIoU-SB'][i] = (1 - small_build_pred_np).astype(np.int32)

        # * Pad gt small buildings
        for pad_buffer in pad_buffers:
            padded_small_build_gt_gpd = pad_small_buildings(
                buildings_gpd=small_build_gt_gpd, pad_buffer=pad_buffer)
            padded_small_build_gt_np = get_nparray_from_gpd(
                padded_small_build_gt_gpd, file_name[i], build_pred.shape[1])

            # Union (predicted small buildings and padded gt small buildings)
            bg_mask_np = ((padded_small_build_gt_np + small_build_pred_np) > 0).astype(np.int32)

            # * Mask predictions with padded gt small buildings and predicted small buildings
            masked_bg_pred = np.multiply(bg_pred[i, :], bg_mask_np)
            masked_bg_gt = np.multiply(bg_gt[i, :], bg_mask_np)

            gt_mask_bg['SmIoU-{}'.format(pad_buffer)][i] = masked_bg_gt
            pred_mask_bg['SmIoU-{}'.format(pad_buffer)][i] = masked_bg_pred

        # Class 1 (building) ---------------------
        # * Mask predictions with small buildings
        small_build_pred_masked_np = np.multiply(build_pred[i, :], small_build_gt_np)

        # * Compute mIOU with adjusted gt/pred
        gt_mask_bd['SmIoU-V2'][i, :] = small_build_gt_np
        pred_mask_bd['SmIoU-V2'][i, :] = small_build_pred_masked_np

        gt_mask_bd['SmIoU-V1'][i, :] = small_build_gt_np
        pred_mask_bd['SmIoU-V1'][i, :] = small_build_pred_np

        gt_mask_bd['mIoU-SB'][i, :] = small_build_gt_np
        pred_mask_bd['mIoU-SB'][i, :] = small_build_pred_np

    return gt_mask_bg, pred_mask_bg, gt_mask_bd, pred_mask_bd
