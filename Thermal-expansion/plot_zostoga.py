# this script plot zostoga emulated by two-layer temperatures and zostoga simulated from GCMs
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

scenarios = ['rcp26','rcp45','rcp85']
model_list = ['MPI-ESM-LR','IPSL-CM5A-LR','HadGEM2-ES','GISS-E2-R','bcc-csm1-1']
year_start=1875
year_end=2299

fig, axes = plt.subplots(5,3,figsize=(15, 20),sharex=True, sharey=True, tight_layout=True)
for m,model in enumerate(model_list):
    f0 = 'output/zostoga_driftcorrection_{:}.nc'.format(model)
    ds0 = xr.open_dataset(f0)
    f1 = 'output/zostoga_FAIR-TLM_allRCP_{:}.nc'.format(model)
    ds1 = xr.open_dataset(f1)
    years = ds1.years

    for i,scen in enumerate(scenarios):
        zostoga0=ds0['zostoga_dc_'+scen].sel(years=slice(year_start,year_end)).values
        zostoga1=ds1['TE_'+scen].sel(years=slice(year_start,year_end)).values

        # compute RMSE of zostoga between GCM and SCM
        rmse = np.sqrt(np.mean((zostoga1-zostoga0)**2))
        print(rmse)

        # plot
        axes[m,i].plot(years, zostoga0, color='b', label='GCM')
        axes[m,i].plot(years, zostoga1, color='r', label='FAIR_TLM(rmse={:5.2f})'.format(rmse))
        axes[m,i].legend(fontsize=16)
        axes[m,i].set_title('{:} {:} '.format(model, scenarios[i]),fontsize=16)
        axes[m,i].set_xlabel('Years',fontsize=16)
        axes[m,i].set_ylabel(r'Thermal expansion (m)',fontsize=16)

fig.savefig('../figures/Thermal_expansion_zostoga_compair_FAIR-TLM_GCM.pdf', dpi=300)
plt.close(fig)
