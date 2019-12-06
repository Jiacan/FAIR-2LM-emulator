# this script compute the GMST and temperature of deep ocean and compare with the GMST in GCM
import fair
from fair.RCPs import rcp3pd, rcp45, rcp85
import numpy as np
import xarray as xr
import os
import matplotlib.pyplot as plt

model_list = ['MPI-ESM-LR','IPSL-CM5A-LR','HadGEM2-ES','GISS-E2-R','bcc-csm1-1']
f = 'parameters/2layer_parameters.txt'
data = np.loadtxt(f, skiprows=1, dtype='str')
model_list1 = data[:,0]
lambda_ = data[:,1].astype('float')
epsilon = data[:,2].astype('float')
C = data[:,3].astype('float')
Co = data[:,4].astype('float')
gamma = data[:,5].astype('float')

f1 = 'parameters/tcrecs.txt'
data1 = np.loadtxt(f1, skiprows=2, dtype='str',delimiter=',')
tcr = data1[:,1].astype('float')
ecs = data1[:,2].astype('float')

# F2XCO2 in CMIP5 models
f2 = 'parameters/F2XCO2_CMIP5.txt'
data2 = np.loadtxt(f2, skiprows=1, dtype='str')
model_list2 = data2[:,0]
F2XCO2 = data2[:,1].astype('float')

# aerosol forcing for scaling
f3 = 'parameters/presentday_aerosol.txt'
data3 = np.loadtxt(f3, skiprows=2, dtype='str',delimiter=',')
af_PD= data3[:,1].astype('float')

# load GMST data for a model
scenarios = ['rcp26','rcp45','rcp85']
time = np.arange(1875, 2301)
for m, model in enumerate(model_list):
    print(model)
    TCR = tcr[m]
    ECS = ecs[m]
    ind0 = np.where(model_list2==model)[0]
    F2x = F2XCO2[ind0]

    # derive T and T0 from the FAIR_2LM
    ind = np.where(model_list1==model)[0]

    C26, F26, T26 = fair.forward_2LM.fair_scm(emissions=rcp3pd.Emissions.emissions,lambda_=lambda_[ind], C_u = C[ind], C_d=Co[ind], gamma = gamma[ind], epsilon = epsilon[ind], tcrecs = np.array([TCR,ECS]), F2x=F2x, scale_aerosol=True, af_PD = af_PD[m])

    C45, F45, T45 = fair.forward_2LM.fair_scm(emissions=rcp45.Emissions.emissions,lambda_=lambda_[ind], C_u = C[ind], C_d=Co[ind], gamma = gamma[ind], epsilon = epsilon[ind], tcrecs = np.array([TCR,ECS]), F2x=F2x, scale_aerosol=True, af_PD = af_PD[m])

    C85, F85, T85 = fair.forward_2LM.fair_scm(emissions=rcp85.Emissions.emissions,lambda_=lambda_[ind], C_u = C[ind], C_d=Co[ind], gamma = gamma[ind], epsilon = epsilon[ind], tcrecs = np.array([TCR,ECS]), F2x=F2x, scale_aerosol=True, af_PD = af_PD[m])

    # compute RMSE between GMST and T
    # index of years for plot
    ind_yr = np.where((rcp85.Emissions.year>=time[0])&(rcp85.Emissions.year<=time[-1]))[0]
    # index of years for baseline
    ind_bl = np.where((rcp85.Emissions.year>=1875)&(rcp85.Emissions.year<=1900))[0]
    T_fair = np.zeros([3,len(ind_yr),2])
    T_fair[0,:,:] = T26[ind_yr,:]-np.tile(np.mean(T26[ind_bl,:],axis=0).T,(len(ind_yr),1))
    T_fair[1,:,:] = T45[ind_yr,:]-np.tile(np.mean(T45[ind_bl,:],axis=0).T,(len(ind_yr),1))
    T_fair[2,:,:] = T85[ind_yr,:]-np.tile(np.mean(T85[ind_bl,:],axis=0).T,(len(ind_yr),1))

    # save data
    ds_out = xr.Dataset({'T_26':('years', T_fair[0,:,0]),
                    'To_26':('years', T_fair[0,:,1]),
                    'T_45':('years', T_fair[1,:,0]),
                    'To_45':('years', T_fair[1,:,1]),
                    'T_85':('years', T_fair[2,:,0]),
                    'To_85':('years', T_fair[2,:,1])},
                    coords={'years': time},
                    attrs={'description': 'surface and deep temperature obtained by FIAR-2LM',
                            'contact':'Jiacan Yuan, jiacan.yuan@gmail.com'})
    ds_out.to_netcdf('output/twolayer_temperature_allRCP_{:}.nc'.format(model),mode='w')
