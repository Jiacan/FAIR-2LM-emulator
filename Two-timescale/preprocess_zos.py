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

    # historical
    f1 = sorted(glob('/projects/kopp/jy519/data/CMIP5/monthly/{:}/zos_Omon_{:}_historical_r1i1p1_*.nc'.format(model.lower(), model)))
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
    years1 = np.arange(yr_temp1.min(), 2006)
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
        zosh_anom = zos_da-zosm
    elif (model == 'GISS-E2-R') | (model == 'HadGEM2-ES' )|(model == 'bcc-csm1-1'):
        zos_da = xr.DataArray(zosh_am, coords=[years1, zosh[lat_name], zosh[lon_name]], dims=['time', 'lat', 'lon'])
        zosm = global_average(zos_da)
        zosh_anom = zos_da-zosm
    # smooth the zos anomaly by a 20-yr moving average
    zosh_sm = zosh_anom.rolling(time=20,center=True).mean()

    zos_anom_sm_list = []
    # rcps
    for i, scen in enumerate(scenarios):
        print(scen)
        f1 = sorted(glob('/projects/kopp/jy519/data/CMIP5/monthly/{:}/zos_Omon_{:}_{:}_r1i1p1_*.nc'.format(model.lower(), model, scen)))
        if len(f1)==1:
            ds1 = xr.open_dataset(f1[0])
            zos = ds1.zos
        else:
            zos_list = []
            for j in range(len(f1)):
                ds1 = xr.open_dataset(f1[j])
                if model=='GISS-E2-R':
                    temp = ds1.zos
                    zos1=temp.where(temp!=0)
                else:
                    zos1 = ds1.zos
                zos_list.append(zos1)
            zos = xr.concat(zos_list, dim='time')

        lat = ds1[lat_name]
        lon = ds1[lon_name]
        nlat = len(lat)
        nlon = len(lon)
        # compute the annual mean
        try:
            yr_temp1 = zos.time.dt.year
            years1 = np.arange(yr_temp1.min(), yr_temp1.max()+1)
            if zos.time.dt.month[0]==12:
                years1=years1[1:]
        except TypeError:
            years1 = range(zos.time[0].dt.year, zos.time[-1].dt.year+1)
        zos_am = np.zeros([len(years1),nlat,nlon])
        for j, year in enumerate(years1):
            if model=='bcc-csm1-1':
                zos_am[j,:,:] = zos.sel(time=str(year).zfill(4)).mean(dim='time')
            else:
                zos_am[j,:,:] = zos.sel(time=str(year)).mean(dim='time')


        # remove global average at each time step
        # convet the zos from np.array to data array
        if (model == 'MPI-ESM-LR')|(model == 'IPSL-CM5A-LR'):
            zos_da = xr.DataArray(zos_am, coords=[years1, zosh.j, zosh.i], dims=['time', 'j', 'i'])
            zosm = global_average_ij(zos_da, ds.lat)
            zos_anom = zos_da-zosm
        elif (model == 'GISS-E2-R') | (model == 'HadGEM2-ES' )|(model == 'bcc-csm1-1'):
            zos_da = xr.DataArray(zos_am, coords=[years1, zos[lat_name], zos[lon_name]], dims=['time', 'lat', 'lon'])
            zosm = global_average(zos_da)
            zos_anom = zos_da-zosm

        # smooth the zos anomaly by a 20-yr moving average
        zosr_sm = zos_anom.rolling(time=20,center=True).mean()
        zos_total = xr.concat((zosh_sm,zosr_sm),dim='time')
        zos_anom_sm_list.append(zos_total)

    zos_anom_sm = xr.concat(zos_anom_sm_list, dim=pd.Index(scenarios, name='scenario'))

    return zos_anom_sm

if __name__ == '__main__':
    main()
