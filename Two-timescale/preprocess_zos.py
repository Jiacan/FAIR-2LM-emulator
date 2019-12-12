# this script remove the global mean from zos and smooth the zos by 20-year moving average

from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import pandas as pd
import sys

# function of global mean for different horizontal coordinary in ocean
def global_average(da, lat_name='lat', lon_name='lon'):
    lat_weights = np.cos(da[lat_name] * np.pi/180.)
    weights = xr.ones_like(da) * lat_weights

    masked_weights = weights.where(~da.isnull())

    normalized_masked_weights = masked_weights / masked_weights.mean(dim=(lat_name, lon_name))

    return (da*normalized_masked_weights).mean(dim=(lat_name, lon_name))

def global_average_ij(da, lat, lat_name='j', lon_name='i'): # two_dimentional lat and lon
    lat_weights = np.cos(lat * np.pi/180.)
    weights = lat_weights

    masked_weights = weights.where(~da.isnull())

    normalized_masked_weights = masked_weights / masked_weights.mean(dim=(lat_name, lon_name))

    return (da*normalized_masked_weights).mean(dim=(lat_name, lon_name))

def zos_preprocessing(model):
    scenarios = ['rcp26','rcp45','rcp85']
    # as different model has diffent variable name for lat and lon,
    # here we specify coordinate name for different models
    if (model == 'MPI-ESM-LR')|(model == 'IPSL-CM5A-LR'):
        lat_name = 'j'
        lon_name = 'i'
    elif (model == 'GISS-E2-R') | (model == 'HadGEM2-ES' ):
        lat_name = 'lat'
        lon_name = 'lon'
    elif (model == 'bcc-csm1-1'):
        lat_name = 'rlat'
        lon_name = 'rlon'

    zos_anom_sm_list = []
    for i, scen in enumerate(scenarios):
        print(scen)
        f1 = sorted(glob('/projects/kopp/jy519/data/CMIP5/monthly/{:}/{:}/zos_Omon_{:}_{:}_r1i1p1_*.nc'.format(scen, model, model, scen)))
        if len(f1)==1:
            ds1 = xr.open_dataset(f1[0])
            zosh = ds1.zos
        else:
            zosh_list = []
            for j in range(len(f1)):
                ds1 = xr.open_dataset(f1[j])
                if model=='GISS-E2-R':
                    temp = ds1.zos
                    zos1=temp.where(temp!=0)
                else:
                    zos1 = ds1.zos
                zosh_list.append(zos1)
            zosh = xr.concat(zosh_list, dim='time')

        lat = ds1[lat_name]
        lon = ds1[lon_name]
        nlat = len(lat)
        nlon = len(lon)
        # compute the annual mean
        yr_temp1 = zosh.time.dt.year
        years1 = np.arange(yr_temp1.min(), yr_temp1.max()+1)
        if zosh.time.dt.month[0]==12:
            years1=years1[1:]
        zosh_am = np.zeros([len(years1),nlat,nlon])
        for i, year in enumerate(years1):
            zosh_am[i,:,:] = zosh.sel(time=str(year)).mean(dim='time')


        # remove global average at each time step
        # convet the zos from np.array to data array
        if (model == 'MPI-ESM-LR')|(model == 'IPSL-CM5A-LR'):
            zos_da = xr.DataArray(zosh_am, coords=[years1, zosh.j, zosh.i], dims=['time', 'j', 'i'])
            zosm = global_average_ij(zos_da, ds.lat)
            zos_anom = zos_da-zosm
        elif (model == 'GISS-E2-R') | (model == 'HadGEM2-ES' )|(model == 'bcc-csm1-1'):
            zos_da = xr.DataArray(zosh_am, coords=[years1, zosh[lat_name], zosh[lon_name]], dims=['time', 'lat', 'lon'])
            zosm = global_average(zos_da)
            zos_anom = zos_da-zosm

        # smooth the zos anomaly by a 20-yr moving average
        zos_sm = zos_anom.rolling(time=20,center=True).mean()
        zos_anom_sm_list.append(zos_sm)
    zos_anom_sm = xr.concat(zos_anom_sm_list, dim=pd.Index(scenarios, name='scenario'))

    return zos_anom_sm

if __name__ == '__main__':
    main()
