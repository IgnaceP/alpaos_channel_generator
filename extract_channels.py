import cv2 as cv
from alpaos_channel_generator.connect_new_channels import *
from shapely.geometry import Polygon
import geopandas as gpd

def extractChannels(alp, epsg = 32717):
    """
    Function to extract the channels from the alpaos object
    :param alp: AlpaosChannelCreator object
    :return: GeoDataFrame with the channels
    """

    Z = connectChannels(alp, region_size = 10)

    chan_mask = ((Z < 2) & ~np.isnan(Z)).astype(np.uint8)
    ret, thresh = cv.threshold(chan_mask, 0, 255, cv.THRESH_BINARY)
    contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    res = alp.meta['resolution']
    TL_x = alp.meta['TL_x']
    TL_y = alp.meta['TL_y']
    
    contours_correct_coords = []
    for contour in contours:
        contour = contour[:,0,:]
        contour[:,0] = TL_x + res/2 + contour[:,0]*res
        contour[:,1] = TL_y - res/2 - contour[:,1]*res
        contours_correct_coords.append(contour)
        

    pol = Polygon(contours_correct_coords[0], contours_correct_coords[1:])
    pol = pol.simplify(1)
    gdf = gpd.GeoDataFrame(geometry=[pol])
    gdf.crs = f'EPSG:{epsg}'

    return gdf
    