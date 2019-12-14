# this script apply pattern scaling on zos through multiple-linear regression on GMST and deep-ocean temperature
import matplotlib
import numpy as np
from preprocess_zos import zos_preprocessing
import matplotlib.pyplot as plt
from scipy.stats import linregress
import xarray as xr
from glob import glob
import statsmodels.api as sm
from statsmodels.formula.api import ols
import pandas as pd
import sys

model='bcc-csm1-1'
# set the start and end years for the target period
year_start = 1981
year_end = 2100

print(model)
# load T and To
print('load temperature data ...')
f = '../FAIR-TLM/output/twolayer_temperature_allRCP_{:}.nc'.format(model)
ds = xr.open_dataset(f)
T_26 = ds.T_26
To_26 = ds.To_26
T_45 = ds.T_45
To_45 = ds.To_45
T_85 = ds.T_85
To_85 = ds.To_85

# pre-processing zos data by removing global mean and smoothing by a 20-year moving average
print('load zos data ...')

# as different model has diffent variable name for lat and lon,
# here we specify coordinate name for different models
if (model == 'MPI-ESM-LR')|(model == 'IPSL-CM5A-LR'):
    lat_name = 'j'
    lon_name = 'i'
elif (model == 'GISS-E2-R') | (model == 'HadGEM2-ES' )|(model == 'bcc-csm1-1'):
    lat_name = 'lat'
    lon_name = 'lon'

# compute smoothed zos
# f2 = 'output/{:}_zosanom_smoothed.nc'.format(model)
# ds2 = xr.open_dataset(f2)
# zos_anom_sm=ds2.zosanom_sm
zos_anom_sm = zos_preprocessing(model)
lat = zos_anom_sm[lat_name]
lon = zos_anom_sm[lon_name]
nlat = len(lat)
nlon = len(lon)


#################################
# two-time scale pattern scaling
#################################
print('pattern scaling ...')
slope2 = np.zeros([2,nlat,nlon])
intercept2 = np.zeros([nlat,nlon])
pvalues2 = np.zeros([2,nlat,nlon])

gmt_sm = xr.concat((T_26.sel(years=slice(year_start, year_end)),T_45.sel(years=slice(year_start, year_end)),T_85.sel(years=slice(year_start, year_end))), dim='years')
To = xr.concat((To_26.sel(years=slice(year_start, year_end)),To_45.sel(years=slice(year_start, year_end)),To_85.sel(years=slice(year_start, year_end))), dim='years')
zos_sm = xr.concat((zos_anom_sm.sel(time=slice(year_start, year_end),scenario=b'rcp26'),zos_anom_sm.sel(time=slice(year_start, year_end),scenario=b'rcp45'),zos_anom_sm.sel(time=slice(year_start, year_end),scenario=b'rcp85')), dim='time')
for n in range(nlat):
    for m in range(nlon):
        temp1 = pd.DataFrame(np.column_stack([zos_sm[:,n,m],To, gmt_sm]), columns = ['zos','To','gmt'])
        try:
            temp2 = ols(formula = "zos ~ To + gmt_sm",
                     data=temp1, missing='drop').fit()
            slope2[0,n,m] = temp2.params.To  # slope coefficient of To
            slope2[1,n,m] = temp2.params.gmt_sm  # slope coefficient of GMST
            intercept2[n,m] = temp2.params.Intercept
        except ValueError:
            slope2[0,n,m] = np.nan
            slope2[1,n,m] = np.nan
            intercept2[n,m] = np.nan

# save data
print('save data to netcdf file ...')
# save slope and intercept
ds1 = xr.Dataset({'slope':(['two_layers','lat','lon'], slope2),
                    'intercept':(['lat','lon'], intercept2)},
                    coords={
                    'two_layers':['deep', 'surface'],
                    'lat': lat.values,
                    'lon': lon.values},
                   attrs={'description': 'slopes, intercept of two-scale pattern scaling of zos on deep and surface temperature obtained from FAIR-TLM',
                         'contact': 'Jiacan Yuan, jiacan.yuan@gmail.com'})

ds1.to_netcdf('output/{:}_slopes_intercept_T_To_allRCP_{:}-{:}.nc'.format(model,year_start, year_end),mode='w')

# save smoothed zos data
# ds_out = xr.Dataset({'zosanom_sm':zos_anom_sm.sel(time=slice(year_start, year_end))},
#                 attrs = {'description': 'zos anomaly (deviating from 1986-2005) smoothed by 20-year running average from {:}'.format(model),'contact': 'Jiacan Yuan, jiacan.yuan@gmail.com'})

# ds_out.to_netcdf('{:}/output/{:}_zosanom_smoothed.nc'.format(Dirt, model),mode='w')
