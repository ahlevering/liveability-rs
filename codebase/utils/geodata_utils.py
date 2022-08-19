import json
from pathlib import Path

from osgeo import gdal
from osgeo import osr
from tqdm import tqdm
import numpy as np
from shapely.geometry import shape, GeometryCollection

def load_geojson_polys(polys_file):
    with open(polys_file) as f:
        features = json.load(f)["features"]
    geoms = GeometryCollection([shape(feature["geometry"]).buffer(0) for feature in features])    
    records = [f['properties'] for f in features]
    return geoms, records

class LBMRasterSegmenter():
    def __init__(self, raster_tile, lbm_polys_file):
        self._set_raster(raster_tile)
        self._set_polys(lbm_polys_file)

        self.RDNEW_OGC_WKT = """PROJCS["Amersfoort / RD New",GEOGCS["Amersfoort",DATUM["Amersfoort",SPHEROID["Bessel 1841",6377397.155,299.1528128,AUTHORITY["EPSG","7004"]],TOWGS84[565.417,50.3319,465.552,-0.398957,0.343988,-1.8774,4.0725],AUTHORITY["EPSG","6289"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AUTHORITY["EPSG","4289"]],PROJECTION["Oblique_Stereographic"],PARAMETER["latitude_of_origin",52.15616055555555],PARAMETER["central_meridian",5.38763888888889],PARAMETER["scale_factor",0.9999079],PARAMETER["false_easting",155000],PARAMETER["false_northing",463000],UNIT["metre",1,AUTHORITY["EPSG","9001"]],AXIS["X",EAST],AXIS["Y",NORTH],AUTHORITY["EPSG","28992"]]"""
    
        # Outputs
        self.file_list = {}

    def _set_raster(self, raster_tile):
        self.raster_tile = gdal.Open(raster_tile)

    def _set_polys(self, polys_file):
        self.geoms, self.records = load_geojson_polys(polys_file)

    def subset_raster_by_lbm_polys(self, xsize, ysize, out_patches_dir, set_name=None, overwrite_patches=False):
        driver = gdal.GetDriverByName("GTiff")
        Path(out_patches_dir).mkdir(parents=True, exist_ok=True)

        # Get xy ranges for raster
        ulx, xres, xskew, uly, yskew, yres  = self.raster_tile.GetGeoTransform()
        ras_x_range = [ulx, ulx + (self.raster_tile.RasterXSize * xres)]
        ras_y_range = [uly + (self.raster_tile.RasterYSize * yres), uly]

        n_pixels_in_xsize = abs(round(xsize * (1/xres)))
        n_pixels_in_ysize = abs(round(ysize * (1/yres)))

        for i, poly in enumerate(tqdm(self.geoms)):
            # Get centroid, round to origin of grid
            poly_x_range, poly_y_range = self._get_offset_range_from_centroid(poly, xsize, ysize)

            in_x_range = poly_x_range[0] > ras_x_range[0] and poly_x_range[1] < ras_x_range[1]
            in_y_range = poly_y_range[0] > ras_y_range[0] and poly_y_range[1] < ras_y_range[1]

            if in_x_range and in_y_range and poly.area > 3500 and not self.records[i]['score'] == None:
                # Create output raster
                grid_id = self.records[i]['id']
                out_filepath = out_patches_dir + str(grid_id) +'.tiff'                
                if overwrite_patches or not Path(out_filepath).exists:
                    out_raster = driver.Create( out_filepath,
                                                xsize=n_pixels_in_xsize,
                                                ysize=n_pixels_in_ysize,
                                                bands=3,
                                                options=["INTERLEAVE=PIXEL"])

                    # Read & write data by offset data relative to top-left
                    x_offset = int(abs(round((poly_x_range[0] - ras_x_range[0]) * (1/xres))))
                    y_offset = int(abs(round((poly_y_range[1] - ras_y_range[1]) * (1/yres))))
                    raster_data = self.raster_tile.ReadAsArray( x_offset, y_offset,
                                                                n_pixels_in_xsize, n_pixels_in_ysize)[:3,:,:]
                    out_raster.WriteRaster(0,0, n_pixels_in_xsize, n_pixels_in_ysize, raster_data.tostring(),
                                                n_pixels_in_ysize, n_pixels_in_ysize, band_list=[1,2,3])            

                    # Set geotransform
                    out_ul = [poly_x_range[0], poly_y_range[1]]
                    out_raster.SetGeoTransform([out_ul[0], xres, xskew, out_ul[1], yskew, yres])
                    
                    # Set projection
                    out_raster.SetProjection(self.RDNEW_OGC_WKT)           
                    
                    out_raster.FlushCache()
                    out_raster = None
                
                ## Clean-up of useless info
                del self.records[i]['gml_id']
                del self.records[i]['gridcode']
                del self.records[i]['id']
                del self.records[i]['scale']

                if set_name:
                    self.records[i]['set_name'] = set_name
                self.records[i]['x'] = float(poly.centroid.xy[0][0])
                self.records[i]['y'] = float(poly.centroid.xy[1][0])
                self.file_list[f'{grid_id}.tiff'] = self.records[i]
    
    def _get_offset_range_from_centroid(self, poly, x_offset, y_offset):
        centroid = poly.centroid.xy
        poly_xmin = centroid[0][0] - centroid[0][0]%100
        poly_ymin = centroid[1][0] - centroid[1][0]%100
        poly_x_range = (poly_xmin, poly_xmin+100)
        poly_y_range = (poly_ymin, poly_ymin+100)
        return poly_x_range, poly_y_range
    
def lookup_grid_scores(polys_file, grid_ids):
    _, records = load_geojson_polys(polys_file)
    grid_scores = {}
    
    for entry in records:
        if entry['gridcode'] in grid_ids:
            grid_scores[str(entry['gridcode'])] = entry
    return grid_scores