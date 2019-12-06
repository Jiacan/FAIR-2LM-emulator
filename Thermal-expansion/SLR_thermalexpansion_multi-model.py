# this script compute the global mean SLR due to thermal expansion for multiple models
# units
# C, Co: W yr /m2/K
# gamma: W/m2/K
# lambda: W/m2/K
# N: W/m2
###########################################

import matplotlib
from scipy.optimize import minimize
import numpy as np
import xarray as xr
from glob import glob
import pandas as pd
from datetime import datetime
import sys

def func(x):
    # compute heat content by integrating with time
    up_26 = T_26*C
    low_26 = To_26*Co
    N_26 = np.zeros(len(years))
    N_26[1:] = up_26[1:]-up_26[:-1] + low_26[1:]-low_26[:-1]
    TE_26 = np.cumsum(N_26*earth_area)* x * 365*24*3.6*1e3*1e-24
    TE_26_anom = TE_26-TE_26[(years_T>=1986)&(years_T<=2005)].mean()

    up_45 = T_45*C
    low_45 = To_45*Co
    N_45 = np.zeros(len(years))
    N_45[1:] = up_45[1:]-up_45[:-1] + low_45[1:]-low_45[:-1]
    TE_45 = np.cumsum(N_45*earth_area) * x * 365*24*3.6*1e3*1e-24
    TE_45_anom = TE_45-TE_45[(years_T>=1986)&(years_T<=2005)].mean()

    up_85 = T_85*C
    low_85 = To_85*Co
    N_85 = np.zeros(len(years))
    N_85[1:] = up_85[1:]-up_85[:-1] + low_85[1:]-low_85[:-1]
    TE_85 = np.cumsum(N_85*earth_area) * x * 365*24*3.6*1e3*1e-24
    TE_85_anom = TE_85-TE_85[(years_T>=1986)&(years_T<=2005)].mean()

    # compute RMSE of zostoga between GCM and SCM
    rmse_26 = np.sqrt(np.mean((TE_26_anom-zostoga_list[0])**2))
    rmse_45 = np.sqrt(np.mean((TE_45_anom-zostoga_list[1])**2))
    rmse_85 = np.sqrt(np.mean((TE_85_anom-zostoga_list[2])**2))

    return (rmse_26+rmse_45+rmse_85)

scenarios = ['rcp26','rcp45','rcp85']
model_list = ['MPI-ESM-LR','IPSL-CM5A-LR','HadGEM2-ES','GISS-E2-R','bcc-csm1-1']
eeh_list = [0.113,0.092,0.112,0.127,0.113]
time = np.arange(1850, 2301)
a = 6.37*1e6
earth_area = 4*np.pi*a**2

# load parameters
f = '../FAIR-TLM/parameters/2layer_parameters.txt'
data = np.loadtxt(f, skiprows=1, dtype='str')
model_list1 = data[:,0]

year_start=1875
year_end=2300
years = np.arange(year_start, year_end)

for m, model in enumerate(model_list):
    print(model)
    eeh = eeh_list[model_list==model]    # expansion efficiency of heat units: m/YJ
    # load drift-corrected zostoga
    f0 = 'output/zostoga_driftcorrection_{:}.nc'.format(model)
    ds0 = xr.open_dataset(f0)
    zostoga_list = []
    for scen in scenarios:
        zostoga_list.append(ds0['zostoga_dc_'+scen].sel(years=slice(year_start,year_end-1)).values)

    # load T and To
    f = '../FAIR-TLM/output/twolayer_temperature_allRCP_{:}.nc'.format(model)
    ds = xr.open_dataset(f)
    T_26 = ds.T_26.sel(years=slice(year_start,year_end-1)).values
    To_26 = ds.To_26.sel(years=slice(year_start,year_end-1)).values
    T_45 = ds.T_45.sel(years=slice(year_start,year_end-1)).values
    To_45 = ds.To_45.sel(years=slice(year_start,year_end-1)).values
    T_85 = ds.T_85.sel(years=slice(year_start,year_end-1)).values
    To_85 = ds.To_85.sel(years=slice(year_start,year_end-1)).values
    years_T = years

    C = data[model_list1==model,3].astype('float')
    Co = data[model_list1==model,4].astype('float')


    # optimize eeh
    x0 = eeh
    res=minimize(func, x0, method='SLSQP')
    eeh = res.x
    print(eeh)

    # compute heat content by integrating with time
    up_26 = T_26*C
    low_26 = To_26*Co
    N_26 = np.zeros(len(years))
    N_26[1:] = up_26[1:]-up_26[:-1] + low_26[1:]-low_26[:-1]
    TE_26 = np.cumsum(N_26*earth_area)*eeh * 365*24*3.6*1e3*1e-24
    TE_26_anom = TE_26-TE_26[(years_T>=1986)&(years_T<=2005)].mean()

    up_45 = T_45*C
    low_45 = To_45*Co
    N_45 = np.zeros(len(years))
    N_45[1:] = up_45[1:]-up_45[:-1] + low_45[1:]-low_45[:-1]
    TE_45 = np.cumsum(N_45*earth_area) *eeh * 365*24*3.6*1e3*1e-24
    TE_45_anom = TE_45-TE_45[(years_T>=1986)&(years_T<=2005)].mean()

    up_85 = T_85*C
    low_85 = To_85*Co
    N_85 = np.zeros(len(years))
    N_85[1:] = up_85[1:]-up_85[:-1] + low_85[1:]-low_85[:-1]
    TE_85 = np.cumsum(N_85*earth_area) *eeh * 365*24*3.6*1e3*1e-24
    TE_85_anom = TE_85-TE_85[(years_T>=1986)&(years_T<=2005)].mean()

    # compute RMSE of zostoga between GCM and SCM
    rmse_26 = np.sqrt(np.mean((TE_26_anom-zostoga_list[0])**2))
    rmse_45 = np.sqrt(np.mean((TE_45_anom-zostoga_list[1])**2))
    rmse_85 = np.sqrt(np.mean((TE_85_anom-zostoga_list[2])**2))
    print(rmse_85, rmse_45, rmse_26)

    # save data
    ds_out = xr.Dataset({'TE_rcp26':('years', TE_26_anom),
                        'TE_rcp45':('years', TE_45_anom),
                        'TE_rcp85':('years', TE_85_anom)},
                    coords={'years': years},
                    attrs={'description': 'Global-mean thermosteric sea level rise emulated by 2-layer temperatures deviating from the baseline period 1986-2005',
                            'contact':'Jiacan Yuan, jiacan.yuan@gmail.com'})
    ds_out.to_netcdf('output/zostoga_FAIR-TLM_allRCP_{:}.nc'.format(model),mode='w')
