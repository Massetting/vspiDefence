# -*- coding: utf-8 -*-
"""
Created on Tue Jul 10 15:05:09 2018

@author: Andrea Massetti

@mail: andrea.massetti@gmail.com

- Read ncdf is a super class to read ncdf files from a folder. The method __load_one was used for testing
- Pixel quality checks the quality of all the pixels contained in a box and produces a list of dates

Edits version 1.1.0:
    Massive changes to job scheduling procedure with dask
Edits version 1.1.1:
    added bulk ndvi
    exports in /short
Edits version 1.1.2:
    replaced rmse with r_sq
Edits on v 1.1.3:
    need fiona and shapely
    load only the area of interest not the entire tile as before.
    the cloud cover percentage is only of the shape area
    cloud cover reduced to 5%
    added standard errors calculation and output in 1 - ncdf att[] and 2 - log file
Edits on v 1.1.3.5: 
    working on pixel quality (cloud cover) because the previous version returned too little dates
"""
__version__ = "1.135"
#%%
import os
import sys
import numpy as np
import xarray as xr
import gdal
from dask.distributed import Client, LocalCluster
import datetime as dt
import argparse
#from pandas import to_datetime 
from dask import delayed as dd
import shapely.geometry as geom 
import fiona
#import pandas as pd
class Read_ncdf():
    """
    Super class to read as xarray the ncdf containing Landsat data
    """
    def __init__(self, folder, keyword=False):
        assert os.path.isdir(folder)
        self.IN_FOLDER = folder
        self.FILES = [os.path.join(self.IN_FOLDER, f) for f in  os.listdir(self.IN_FOLDER) if f.endswith(".nc")]
#        self.keyword = keyword
        if keyword:
            self.shape_path = "/g/data3/xg9/DE/contours/{}.shp".format(keyword) #TODO! hardcoded
        else:
            self.shape_path = False
    def load(self, folder):
        """
        returns a xarray of all the nc files available
        """
#        if WIN:
#            return xr.open_mfdataset(f"{folder}\*.nc",chunks={"time":5})#"x":20,"y":20})
#        elif not WIN:
        
        g = xr.open_mfdataset(f"{folder}/*.nc", chunks={"time":30,'x':500,'y':500}, parallel=True)
        g = g.sortby(g.time)
        if self.shape_path:
            bounds = self.get_shp_extent(self.shape_path)
            temp_1 = g.sel(x=bounds[0], y=bounds[1], method="nearest")
            temp_2 = g.sel(x=bounds[2], y=bounds[3], method="nearest")
            refined_bounds=[]
            refined_bounds.append(float(temp_1.x.values))
            refined_bounds.append(float(temp_1.y.values))
            refined_bounds.append(float(temp_2.x.values))
            refined_bounds.append(float(temp_2.y.values))
            g = g.sel(x=slice(refined_bounds[0], refined_bounds[2]), y=slice(refined_bounds[3], refined_bounds[1]))  
        else:
            pass
        return g
    def get_shp_extent(self, shape_path):
        w = [i for i in fiona.open(str(shape_path))]
        for j in range(len(w)):
            z=w[j]['geometry']
            if z["type"]=='MultiPolygon':
    #            lines = [geom.LineString(l[0]) for l in z['coordinates']]
                bounds = geom.MultiLineString([l[0] for l in z['coordinates']]).bounds
            elif z["type"]=='Polygon':
    #            lines = [geom.LineString(l) for l in z['coordinates']]
                bounds = geom.MultiLineString([l for l in z['coordinates']]).bounds
        return bounds

    def __load_one(self, file_path):
        """
        For now this is used only in the test class, but might come in handy later on
        """
        return xr.open_dataset(file_path)

class Pixel_Quality(Read_ncdf):
    """
    Passes through the pixel quality of Landsat netcdf and outputs 
    the general quality in a scene, provided that the quality is maximum 
    in a bounding box around the study area.
    """
    def __init__(self, folder, keyword, x_in, x_pix, y_in, y_pix):
        self.IN_FOLDER = folder
        self.keyword = keyword
        super(Pixel_Quality, self).__init__(self.IN_FOLDER, self.keyword)
        self.ds = self.load(self.IN_FOLDER)
        self.x_in = x_in
        self.x_pix = x_pix
        self.y_in = y_in
        self.y_pix = y_pix
    def scroll_dates(self, variable, cloud):
        self.x_len = self.x_pix*25
        self.x_fin = self.x_in + self.x_len #+ because w to e
        self.y_len = self.y_pix * 25
        self.y_fin = self.y_in + self.y_len
        print(f"""
        |----------------------------|
        |          {self.y_fin}          |
        |{self.x_in}            {self.x_fin}  |
        |          {self.y_in}          |
        |----------------------------|
        """)
#        vr = variable.sel(y=np.arange(self.y_in, self.y_fin, 25), x=np.arange(self.x_in, self.x_fin,25), method='nearest')
        vr = variable.sel(y=np.arange(self.y_in, self.y_fin, 25), x=np.arange(self.x_in, self.x_fin,25), method='nearest')
        n_good = (vr == 16383).sum(dim=("x","y"))
        g = n_good.values
        tt = n_good.time.values
        gu=[]
        print("DATE, QUALITY\n")
        
        for n, v in enumerate(g):#date_printable, pixel_printable in zip(n_good.time.values, g):
            if v == self.x_pix * self.y_pix:
                gu.append(n)
#            if pixel_printable == self.x_pix * self.y_pix:
                print(f"{tt[n]}, {v} <------ X")
            else:
                print(f"{tt[n]}, {v}")
#        gu = [n for n, v in enumerate(g) if v == self.x_pix*self.y_pix]
        print(f"{dt.datetime.now()}: {len(gu)} good quality dates in the bbox ")
        self.arr = variable.isel(time=gu)#TODO! add here the bounding of the shape!

#        print(f"original chunks: {self.arr.chunks}")
#        a1 = dd((self.arr == 4095).sum(dim=("x","y")).values)
#        
#        a2 = dd((self.arr == 13311).sum(dim=("x","y")).values)
#        a3 = dd((self.arr == 15359).sum(dim=("x","y")).values)
#        a4 = dd((self.arr == 14335).sum(dim=("x","y")).values)
#        a5 = dd((self.arr == 8191).sum(dim=("x","y")).values)
        a1 = dd((self.arr == 16383).sum(dim=("x","y")).values)
        a6 = dd(np.isfinite(self.arr.values).sum(axis=(self.arr.get_axis_num("x"),self.arr.get_axis_num("y"))))
        
#        a7 = dd(((a1+a2+a3+a4+a5) / a6))
        a7 = dd(a1/a6)
        #total = client.compute(a7)
        #j=total.result()
        
        a7 = a7.compute()
        return [(self.arr.time[n]).values for n, v in enumerate(a7) if v > (1-cloud)] #TODO! hardcoded cloudcover
        
        
#        a1 = dd((self.arr == 4095).sum(dim=("x","y")).values)
#        a2 = dd((self.arr == 13311).sum(dim=("x","y")).values)
#        a3 = dd((self.arr == 15359).sum(dim=("x","y")).values)
#        a4 = dd((self.arr == 14335).sum(dim=("x","y")).values)
#        a5 = dd((self.arr == 8191).sum(dim=("x","y")).values)
#        a6 = dd(np.isfinite(self.arr.values).sum(axis=(self.arr.get_axis_num("x"),self.arr.get_axis_num("y"))))
#        #a6b = dd(a6)
#        a7 = dd(((a1+a2+a3+a4+a5) / a6).values)
#        total = client.persist(a7)
#        j=total.result()
##        a1 = (self.arr == 4095).sum(dim=("x","y"))
##        a2 = (self.arr == 13311).sum(dim=("x","y"))
##        a3 = (self.arr == 15359).sum(dim=("x","y"))
##        a4 = (self.arr == 14335).sum(dim=("x","y"))
##        a5 = (self.arr == 8191).sum(dim=("x","y"))
##        a6b = np.isfinite(self.arr.values)
##        a6c=a6b.sum(axis=(self.arr.get_axis_num("x"),self.arr.get_axis_num("y")))
##        a7 = ((a1+a2+a3+a4+a5) / a6c).values
#        return [(self.arr.time[n]).values for n, v in enumerate(a7) if v < cloud]
#        for time in vr.time:
#            part_arr = vr.sel(time=time.values)
##            part_arr = arr.sel(y=np.arange(self.y_in, self.y_fin, 25), x=np.arange(self.x_in, self.x_fin,25), method='nearest')
#            good_pixels = part_arr.where(part_arr==16383).count().values #good px in the subset
#            if good_pixels:
#                if good_pixels / (self.x_pix * self.y_pix) != 1:#percent of good pix in the subset
#                    good_pixels = None
#            if good_pixels:
#                is_data = "dummy"#arr.count()
#                is_good_data = "dummy"#arr.where(arr==16383).count()
#                is_cloud_data = "dummy"#arr.where(arr==13311).count() + arr.where(arr==15359).count() + arr.where(arr==14335).count() + arr.where(arr==8191).count()
#                is_shadow_data = "dummy"#arr.where(arr==4095).count()
#                bad_pix_ratio = "dummy"#(is_shadow_data + is_cloud_data) / is_good_data
#                print(f"{dt.datetime.now()}: processed date {time.values}\n")
#                times.append([time.values, is_data, is_good_data, is_cloud_data, is_shadow_data, bad_pix_ratio])
#        return times

class SWIR_Calculation(Read_ncdf):
    """
    Calculates the linear regression of the swir bands of the forest within 
    a bounding box and computes the VSPI of the temporal cube in a netcdf. 
    Saves a netcdf containing a VSPI map for each Landsat acquisition.
    """
    def __init__(self, folder, keyword, times, calcVSPI, cloud=1, aspect=None, slope=None, intercept=None, x_in=None, x_pix=None, y_in=None, y_pix=None):
        self.IN_FOLDER = folder
        self.keyword = keyword
        super(SWIR_Calculation, self).__init__(self.IN_FOLDER, self.keyword)
        self.aspect = None
        self.cloud_threshold = cloud
        if aspect:
            self.aspect_file = aspect
        self.ds= self.load(self.IN_FOLDER)
        self.times = times
#        if type(times) == list:
#            self.times = [f[0]for f in times if f[5]<=self.cloud_threshold]
#        if type(times) == np.ndarray:#this avoids to recompute the qual_dates. Cloud threshold is handled outside
#            self.times = times
#        print(self.times)
        self.match_dt()
        self.calcVSPI = calcVSPI
        if not self.calcVSPI:
            self.slope = slope
            self.intercept = intercept
        if self.calcVSPI:
            if x_in:
               self.x_left = x_in
               self.x_pix = x_pix
               self.y_bottom = y_in
               self.y_pix = y_pix
            self.compute_average()
            self.lin_regr()
    def match_dt(self):
        self.time_slice = [t for t in self.times]####remember ...t.values for t... here
        try:
            self.ds = self.ds.sel(time=self.time_slice)
        except:
            _, index = np.unique(self.ds['time'], return_index=True)
            self.ds = self.ds.isel(time=index)
            self.ds = self.ds.sel(time=self.time_slice)
    def compute_average(self):
        self.x_len = self.x_pix*25
        self.x_right = self.x_left+self.x_len #+ because w to e
        self.y_len = self.y_pix * 25
        self.y_top = self.y_bottom + self.y_len
        self.subset_b5 = self.ds.swir1.sel(y=np.arange(self.y_bottom, self.y_top,25),x=np.arange(self.x_left,self.x_right,25),method='nearest')
        self.subset_b7 = self.ds.swir2.sel(y=np.arange(self.y_bottom, self.y_top,25),x=np.arange(self.x_left,self.x_right,25),method='nearest')
        self.subset_b3 = self.ds.red.sel(y=np.arange(self.y_bottom, self.y_top,25),x=np.arange(self.x_left,self.x_right,25),method='nearest')
        self.subset_b4 = self.ds.nir.sel(y=np.arange(self.y_bottom, self.y_top,25),x=np.arange(self.x_left,self.x_right,25),method='nearest')        
        self.ndvi = (self.subset_b4 - self.subset_b3)/(self.subset_b4 + self.subset_b3)
        self.ndvi_climatology = self.ndvi.mean(dim="time", skipna=True)
        z = self.ndvi_climatology.values
#        print(z)
        print("NDVI climatology \nSTD:{} \nMEAN{}\nUSING THE FIX VALUE OF 0.4".format(z.std(),z.mean()))
        
        self.cover_subsets_ndvi()
        self.mu_b5 = np.nanmean(self.subset_b5)
        self.mu_b7 = np.nanmean(self.subset_b7)
    def treecover_clip(self, x, y):
        ds = gdal.Open(self.treecover_file)
        gt = ds.GetGeoTransform()
        x_corn =  int((x -gt[0]) / gt[1])
        y_corn = int((y - gt[3]) / gt[5]) #deepest
        self.treecover = ds.ReadAsArray(x_corn, y_corn, self.x_pix, self.y_pix)
    def aspect_clip(self,x,y):
        ds = gdal.Open(self.aspect_file)
        gt = ds.GetGeoTransform()
        x_corn =  int((x -gt[0]) / gt[1])
        y_corn = int((y - gt[3]) / gt[5]) #deepest
        self.aspect = ds.ReadAsArray(x_corn, y_corn, self.x_pix, self.y_pix)
    def cover_subsets_ndvi(self):
        self.subset_b5 = self.subset_b5.where((self.ndvi_climatology > 0.4))
        self.subset_b7 = self.subset_b7.where((self.ndvi_climatology > 0.4))
        print("TOTAL NUMBER OF PIXELS USED TO COMPUTE VSPI:      {}".format((self.ndvi_climatology > 0.4).sum(dim=("x","y")).values))
    def lin_regr(self):
        d_b5 = self.subset_b5 - self.mu_b5
        d_b7 = self.subset_b7 - self.mu_b7
        ssq5 = d_b5**2
        ssq7 = d_b7**2
        self.n_5 = np.sum(~np.isnan(ssq5.values))
        self.n_7 = np.sum(~np.isnan(ssq7.values))
        var5 = np.nansum(ssq5.sum(dim="time"))/(self.n_5 - 1)
        var7 = np.nansum(ssq7.sum(dim="time"))/(self.n_7 - 1)
        ww = d_b5 * d_b7
        cov_5_7 = np.nansum(ww.sum(dim="time"))/(np.sum(~np.isnan(ww.values))-1)
        self.slope = cov_5_7/var5
        self.intercept = self.mu_b7-(self.slope * self.mu_b5)
        
        self.s_err_model = np.sqrt(np.nansum((self.subset_b7 - ((self.subset_b5 * self.slope) + self.intercept))**2) / (self.n_7 - 2))
        self.s_err_intercept = np.sqrt(1 + ((self.mu_b5**2)/var5)) * (self.s_err_model / np.sqrt(self.n_7))
        self.s_err_slope = (1 / (np.sqrt(var5))) * (self.s_err_model / np.sqrt(self.n_7))        
        self.r_sq = cov_5_7 / (np.sqrt(var5) * np.sqrt(var7))

    def calc_vspi(self):
        self.vspi = 1 / np.sqrt((self.slope**2)+1) * (self.ds.swir2 - (self.slope*self.ds.swir1) - self.intercept)
        self.populate_metadata()
        return self.vspi
        
    def populate_metadata(self):
        self.vspi.name = "VSPI"
        att = self.ds.attrs.copy()        
        att["inherited_meaning"] = 'the flag inherited means that the attribute in the list was copied from the lower level collection from Geoscience Australia (GA ARG25)'
        att["acknowledgment_inherited"] = att["acknowledgment"]
        del att["acknowledgment"]
        try:
            att["comment_inherited"] = att["comment"]
            del att["comment"]
        except:
            pass
        att['geospatial_bounds_inherited'] = att['geospatial_bounds']
        del att['geospatial_bounds']
        att['geospatial_bounds_crs_inherited'] = att['geospatial_bounds_crs']
        del att['geospatial_bounds_crs']
        att['geospatial_lat_min_inherited'] = att['geospatial_lat_min']
        del att['geospatial_lat_min']
        att['geospatial_lat_max_inherited'] = ['geospatial_lat_max']
        del att['geospatial_lat_max']
        att['geospatial_lat_units_inherited'] = att['geospatial_lat_units']
        del att['geospatial_lat_units']
        att['geospatial_lon_min_inherited'] = att['geospatial_lon_min']
        del att['geospatial_lon_min']
        att['geospatial_lon_max_inherited'] =att['geospatial_lon_max'] 
        del att['geospatial_lon_max']
        att['geospatial_lon_units_inherited'] = att['geospatial_lon_units']
        del att['geospatial_lon_units']
        att['instrument_inherited'] = att['instrument']
        del att['instrument']
        att['platform_inherited'] = att['platform']
        del att['platform']
        att['cdm_data_type_inherited'] = att['cdm_data_type']
        del att['cdm_data_type']
        att['product_suite_inherited'] = att['product_suite']
        del att['product_suite']
        del att['license'] 
        del att["Conventions"]
        del att["institution"]
        del att["history"]
        del att["keywords_vocabulary"]
        del att["publisher_email"]
        del att['publisher_name']
        del att["publisher_url"]
        att['title'] = "VSPI"
        att["Date_created"] = str(dt.datetime.now())#?
        att["Author e-mail"] = "andrea.massetti@gmail.com"
        att["Citation"] ="""Andrea Massetti, Christoph Rüdiger, Marta Yebra, James Hilton, The Vegetation Structure Perpendicular Index (VSPI): A forest condition index for wildfire predictions,
        Remote Sensing of Environment, Volume 224, 2019, Pages 167-181, ISSN 0034-4257, https://doi.org/10.1016/j.rse.2019.02.004."""
        att["Link to article"] = "https://www.sciencedirect.com/science/article/pii/S0034425719300586"
        att["keywords"] = ["Landsat", "Forest condition", "Wildfire spread", "Vegetation recovery"]
        att["Abstract"] = """Wildfires are a major natural hazard, causing substantial damage to infrastructure as well as being a risk to lives and homes. An understanding of their progression and behaviour is necessary to reduce risks and to develop operational management strategies in the event of an active fire. Many empirical fire-spread models have been developed to predict the spread and overall behaviour of a wildfire, based on a range of parameters such as weather and fuel conditions. However, these parameters may not be available with sufficient accuracy or spatiotemporal resolution to provide reliable fire spread predictions. Fuel condition data include variables such as vegetation quantity, structure and moisture content and, in the event of previous wildfires, the burn severity and stage of ecosystem recovery. In this study, an index called the Vegetation Structure Perpendicular Index (VSPI) is introduced. The VSPI utilises the short-wave infrared reflectance in bands centred at 1.6 and 2.2 μm, essentially representing the amount and structure of the vegetation's woody biomass (as opposed to the photosynthetic activity and moisture content). The VSPI is quantified as the divergence from a linear regression between the two bands in a time series and represents vegetation disturbance and recovery more reliably than indices such as the Normalised Burn Ratio (NBR) and Normalised Difference Vegetation Index (NDVI). The VSPI index generally shows minor inter-annual variability and stronger post-wildfire detection of disturbance over a longer period than NBR and NDVI. The index is developed and applied to major wildfire events within eucalypt forests throughout southern Australia to estimate both burn severity and time to recovery. The VSPI can provide an improved information layer for fire risk evaluation and operational predictions of wildfire behaviour.
        """
        att["Version"] = "Created with code version {}".format(__version__)
        att["Vegetation_line_slope"] = self. slope
        att["Vegetation_line_intercept"] = self.intercept
        try:
            att["Vegetation_line_R_sq"] = self.r_sq
            att["Vegetation_line_linear_model_standard_error"] = self.s_err_model
            att["Vegetation_line_intercept_standard_error"] = self.s_err_intercept
            att["Vegetation_line_slope_standard_error"] = self.s_err_slope
            att["Sample_size"] = self.n_7
            att["Vegetation_line_x_left"] = self.x_left
            att["Vegetation_line_x_right"] = self.x_right
            att["Vegetation_line_y_top"] = self.y_top
            att["Vegetation_line_y_bottom"] = self.y_bottom
            att["Vegetation_line_hor_px"] = self.x_pix
            att["Vegetation_line_ver_px"] = self.y_pix

        except:
            att["Vegetation_line_R_sq"] = "NA"
            att["Vegetation_line_x_left"] = "NA"
            att["Vegetation_line_x_right"] = "NA"
            att["Vegetation_line_y_top"] = "NA"
            att["Vegetation_line_y_bottom"] = "NA"
            att["Vegetation_line_hor_px"] = "NA"
            att["Vegetation_line_ver_px"] = "NA"            
        

#        att["Vegetation_line_dates"] = to_datetime(b.times).sort_values()
        att["Vegetation_line_min_quality"] = self.cloud_threshold
        self.vspi.attrs = att

def GAtile(x_tgt, y_tgt):
    """
    Convert coordinates in australia albers in GA tiles
    """
    print("Fetching GA tile for {}, {}".format(x_tgt, y_tgt))
    def x(tgt):
        for t_x in np.arange(-21, 26):
            x0 = +100000
            x2 = x0 + 99999.83333333 * t_x
            x1 = x2 - 100000
            if x1<tgt<x2:
                if (x2-tgt<250) | (x1-tgt> -250):
                    print("WARNING the x coordinate of the bounding box is close to the tile border: <6250m")
                return t_x
    def y(tgt):
        for t_y in np.arange(-45, -10):
            y0 = 0
            y1 = y0 + 99999.83333333 * t_y
            y2 = y1 + 100000
            if y1<tgt<y2:
                if (y2 - tgt < 250) | (y1 - tgt > -250):
                    print("WARNING the y coordinate of the bounding is close to the tile border: <6250m")
                return t_y
    
    # =============================================================================
    # _, lat1 = GDA94(x1, y1)
    # long1, _ = GDA94(x1, y1)
    # _, lat2 = GDA94(x2, y2)
    # long_2, _= GDA94(x2, y1)
    # =============================================================================
    return f"{x(x_tgt)}_{y(y_tgt)}"
    
#%%
def create_folder(flds):
    for fld in flds:
        if not os.path.isdir(fld):
            print("{} does not exist".format(fld))
            os.makedirs(fld)
            print("-> CREATED")    
if __name__ == "__main__":

#%% parser
    parser = argparse.ArgumentParser(prog="CalcVSPI", description=
                                     'Computes the VSPI for a given tile of Landsat')
    
    parser.add_argument('-n', dest="keyword", default=None, type=str,
                        help='Provide a name for the location you are calculating VSPI. note this keyword must match the shapefile')
    
    parser.add_argument('-s', dest="sensor", choices=["5", "7", "8", "all"], default=5, 
                        metavar='sens', type=str,
                        help='Landsat sensor 5, 7 or 8, for TM, ETM+ and OLI')
#    parser.add_argument('-t', dest="tile", default="12_-42", metavar='tile', type=str,
#                        help='tile according to geoscience australia datacube example: "12_-42"')

    parser.add_argument('--calc', dest="calcVSPI", action='store_true',
                        help='''flag for computing the VSPI's slope and intercept values. 
                        NOTE this always does multiprocessing''')
    parser.add_argument('--NDVI', dest="calcNDVI", action='store_true',
                        help='''flag for computing the VSPI's slope and intercept values. 
                        NOTE this always does multiprocessing''')
    parser.add_argument('--no-calc', dest='calcVSPI', action='store_false')

    parser.add_argument('--vege-line', dest="vege_arg", default="auto", type=str,
                        help='''Provide the vegetation line parameters. 
                        This can be either the fullpath of the text file 
                        containing the vegetation line parameters or a string 
                        containing the slope and intecept separated by a space. 
                        Example 1: --vege_line "0.65 -250"
                        Example 2: --vege_line "C:\\vlparams\vege.txt"''')
    
#    parser.add_argument('--cluster', dest="cluster", choices=[True, False], default=False, type=bool, 
#                        help='True goes multiprocessing, False: single. Default False')
    parser.add_argument('--bbox', dest="bbox", type=int, nargs=4 ,
                        help="""provide 4 integers that represent: 
                            x_coordinate left, number of horizontal pixels,
                            y_coordinate bottom, number of veritcal pixels.
                            example :
                                --bbox 1210311 80 -4146678 60
                                1370686 100 -4193086 100
                            Note: this works to determine the cloud free area 
                            AND the forest used to calculate the vegetation line,
                            if this calculation requested with --calc True
                            """)
    parser.add_argument('--cores', dest="cores",default = 8, type=int,
                        help="cores used in qsub")
    parser.add_argument('--memory', dest="tot_mem",default = 32, type=int,
                        help="total memory available")
    parser.add_argument('--version', action='version', version=__version__)

    jj = parser.parse_args()
    sensor = jj.sensor#8
    cores = jj.cores
    tot_mem = jj.tot_mem
    wk = int(cores/2)
    th = 2
    mem = "{}Gb".format(int(tot_mem/wk)-1)
    keyword = jj.keyword
    #select scenes that have cloud+shadow/total < than threshold
    cloud_cover_threshold = 0.07
    calcVSPI = jj.calcVSPI#False
    try:
        calcNDVI = jj.calcNDVI
    except:
        calcNDVI = False    
    x_in = jj.bbox[0]#1210311
    x_pix = jj.bbox[1]#80
    y_in = jj.bbox[2]#-4146678
    y_pix = jj.bbox[3]#60    
    location = GAtile(x_in, y_in) #jj.tile#"12_-42"
    vege_arg = jj.vege_arg
#%% folders

    IN = "/g/data2/rs0/datacube/002"
#    TC_FILE = "/short/xg9/ama565/treecover/treecoverVIC.tif"       # ISSUE #5
        
    if sensor== "all":
        sensors = ["5", "7", "8"]
    else:
        sensors = [sensor]
    print(f"Calc VSPI working version {__version__} ")
    print(f"sensor: {sensor}\n tile: {location}\n Study area: {keyword}\n Maximum clouds {cloud_cover_threshold}\n Calculate Vege Line{calcVSPI}\n Site location \n{x_in}\n{x_pix}\n{y_in}\n{y_pix}\n")
    print(f"{dt.datetime.now()}: cluster init")

#%% cluster init

    print(f"local cluster init\nCORES:{cores}\nWORKERS:{wk}\n MEM/WORKER:{mem}")
    cluster = LocalCluster(n_workers=wk, threads_per_worker=th, memory_limit=mem)
    client = Client(cluster)
#%% iterate over sensors
    for sensor in sensors:
        if int(sensor) == 7:
            sensor_string = "LS7_ETM_"
        if int(sensor) == 5:
            sensor_string = "LS5_TM_"
        if int(sensor) == 8:
            sensor_string = "LS8_OLI_"
#        print(f"processing sensor {sensor_string}")
        print(f"{dt.datetime.now()}: started processing sensor {sensor}\n ")
        PQ_FOLDER = os.path.join(IN,f"{sensor_string}PQ{os.sep}{location}")
        REFL_FOLDER = os.path.join(IN,f"{sensor_string}NBART{os.sep}{location}")#"/short/xg9/ama565/Landsat/LS7_ETM_NBART/-15_-37"
        OUTFLD_base = "/g/data3/xg9/DE/vspi" #TODO! changed to short!
        OUTFLD = os.path.join(OUTFLD_base, keyword)
        OUTFLDlogs = os.path.join(OUTFLD_base,"logs")
        OUTFLDVL = os.path.join(OUTFLD_base,"vegetation_lines")
        create_folder([OUTFLD, OUTFLDlogs, OUTFLDVL])
        if calcNDVI:
            OUTFLDndvi = os.path.join(OUTFLD, "ndvi")
            create_folder([OUTFLDndvi])
        
        print(f"{dt.datetime.now()}: WORKING ON\ sensor: {sensor}\n GA tile: {location}\n site denom: {keyword}\n cloud cover allowed: {cloud_cover_threshold}\n{calcVSPI}\n box: {x_in}\n{x_pix}\n{y_in}\n{y_pix}\n pixel quality from: {PQ_FOLDER}\n reflectance from: {REFL_FOLDER}\n output at: {OUTFLD}\n logs at: {OUTFLDlogs}\n vegetation line output at: {OUTFLDVL}\n")
#%%load vege parameters
        if not calcVSPI:
            if vege_arg == "auto":
                print(f"{dt.datetime.now()}: vegetation line not specified trying to infer file location")
                try:
                    with open(os.path.join(OUTFLDVL, f"LS5_{location}_{keyword}_cloud_lt{int(cloud_cover_threshold*100)}pc_VL_params.csv")) as txt:
                        z = txt.read()
                    print("successfully retrieved vege line file")
                except:
                    print("please provide vege line parameters or set calcVSPI=True")
                    
            elif os.path.isfile(vege_arg):
                print(f"{dt.datetime.now()}: importing vege line file as specified")
                with open(vege_arg) as txt:
                    z = txt.read()
            else:
                print("{dt.datetime.now()}: using given vege line parameters")
                z = vege_arg
                
            try:
                slope, intercept = [float(f) for f in z.split(" ")]
                print(slope, intercept)
            except:
                print("please provide vege line parameters or set calcVSPI=True")
                j="dummy"
                while j!= "y" or j != "n":
                    j = input("do you want to calculate vege line parameters? y/n")
                if j=="n":
                    sys.exit()
                if j=="y":
                    calcVSPI = True

            
#%%        calculate dates
        if not os.path.isfile(os.path.join(OUTFLDlogs,f"LS{sensor}_{location}_{keyword}_cloud_lt{int(cloud_cover_threshold*100)}pc_DATES.npy")):
            print(f"{dt.datetime.now()}: dates list unavailable, re-computing")
            print(f"{dt.datetime.now()}: started selecting dates\n ")
#            when no bbox?

            if GAtile(x_in, y_in) != GAtile(x_in + x_pix*25, y_in + y_pix*25):
                print("the bounding box is across two tiles. Calling a system exit")
                sys.exit()
#%% do it
            a = Pixel_Quality(PQ_FOLDER, keyword=keyword, x_in=x_in, x_pix=x_pix, y_in=y_in, y_pix=y_pix)
            #TODO!

            times = a.scroll_dates(a.ds.pixelquality, cloud_cover_threshold)
                
# =============================================================================

# =============================================================================
            print(f"{dt.datetime.now()}: obtained {len(times)} dates")
            times = np.array(times)
            np.save(os.path.join(OUTFLDlogs,f"LS{sensor}_{location}_{keyword}_cloud_lt{int(int(cloud_cover_threshold*100))}pc_DATES.npy"), times)
            print(f"{dt.datetime.now()}: dates saved successfully at", os.path.join(OUTFLDlogs,f"LS{sensor}_{location}_{keyword}_cloud_lt{int(int(cloud_cover_threshold*100))}pc_DATES.npy"))
#            only_good_dates = np.array([t[0] for t in times if t[5]<cloud_cover_threshold])
#            np.save(os.path.join(OUTFLDlogs,f"LS{sensor}_{location}_{keyword}_clc_lt{int(cloud_cover_threshold*100)}pcDATES"), only_good_dates)
            
# =============================================================================
#            SHould not be used as the other qual stats are in the process of being discarded
# #            with open(os.path.join(OUTFLDlogs,f"LS{sensor}_{location}_{keyword}_Qual_stats.csv"), "w") as txt:
# #                txt.write("date, is_data, is_good_data, is_cloud_data, is_shadow_data, bad_pix_ratio\n")
# #                for t in times:
# #                    txt.write(f"{t[0]},{t[1]},{t[2]},{t[3]},{t[4]},{t[5]}\n")
# =============================================================================
        elif os.path.isfile(os.path.join(OUTFLDlogs,f"LS{sensor}_{location}_{keyword}_cloud_lt{int(cloud_cover_threshold*100)}pc_DATES.npy")):
            times = np.load(os.path.join(OUTFLDlogs,f"LS{sensor}_{location}_{keyword}_cloud_lt{int(cloud_cover_threshold*100)}pc_DATES.npy"))
            print("dates successfully loaded from",os.path.join(OUTFLDlogs,f"LS{sensor}_{location}_{keyword}_cloud_lt{int(cloud_cover_threshold*100)}pc_DATES.npy"))
        print(f"{dt.datetime.now()}: finished selecting dates\n ")
#%% calculate VSPI 
        if len(times)==0:
            break
        if calcVSPI:
            print(f"{dt.datetime.now()}: started calculating vege line\n ")
            #TODO!

            b = SWIR_Calculation(REFL_FOLDER, keyword, times, calcVSPI, cloud = cloud_cover_threshold, x_in=x_in, x_pix=x_pix, y_in=y_in, y_pix=y_pix)
# =============================================================================

            with open(os.path.join(OUTFLDlogs, f"LS{sensor}_{location}_{keyword}_cloud_lt{int(cloud_cover_threshold*100)}pc_VLstats.csv"), "a") as txt:
                txt.write(f"LS{sensor}_{location} bottom left corner\n long: {x_in} \n lat: {y_in} \n x_size: {x_pix} \n y_size: {y_pix}\n")
                txt.write(f"slope = {b.slope}, intercept = {b.intercept}, R_sq = {b.r_sq}, Vegetation_line_st_err = {b.s_err_model}, Intercept_st_err = {b.s_err_intercept}, Slope_st_err = {b.s_err_slope}, sample size={b.n_7}\n")
                txt.write(f"number of scenes used: {len(times)}\n")
            with open(os.path.join(OUTFLDVL, f"LS{sensor}_{location}_{keyword}_cloud_lt{int(cloud_cover_threshold*100)}pc_VL_params.csv"), "a") as txt:
                txt.write(f"{b.slope} {b.intercept}")
            print(f"{dt.datetime.now()}: finished calculating vege line\n ")
            print(f"{dt.datetime.now()}: started calculating and saving vspi\n ")
            #TODO!

            vspi= b.calc_vspi()
            vspi.to_netcdf(os.path.join(OUTFLD, f"LS{sensor}_{location}_{keyword}_cloud_lt{int(cloud_cover_threshold*100)}pc_VSPI.nc"))

# =============================================================================
            print(f"{dt.datetime.now()}: started calculating and saving vspi\n ")
            
        if not calcVSPI:
            print(f"{dt.datetime.now()}: started calculating and saving vspi\n ")
            b = SWIR_Calculation(REFL_FOLDER, keyword, times, calcVSPI, cloud=cloud_cover_threshold, slope=slope, intercept=intercept)
            #TODO!

            vspi = b.calc_vspi()
            print(f"{dt.datetime.now()}: finished calculating vspi\n ")
            vspi.to_netcdf(os.path.join(OUTFLD, f"LS{sensor}_{location}_{keyword}_cloud_lt{int(cloud_cover_threshold*100)}pc_VSPI.nc"))

# =============================================================================
            print(f"{dt.datetime.now()}: finished saving vspi\n ")
            print("Successfully created vspi stack")
        if calcNDVI:
            ndvi = (b.ds.nir - b.ds.red)/(b.ds.nir + b.ds.red) 
            print(f"{dt.datetime.now()}: finished calculating ndvi\n ")
            ndvi.to_netcdf(os.path.join(OUTFLDndvi, f"LS{sensor}_{location}_ndvi.nc"))
            print(f"{dt.datetime.now()}: finished saving ndvi\n ")
        print(f"{dt.datetime.now()}: finished processing sensor {sensor_string}\n ")
        calcVSPI = False #this should avoid to calculate the vege line for sensor 7 and 8 after they were calculated for sens 5
