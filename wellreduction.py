# -*- coding: utf-8 -*-
"""
Created on Fri Sep 12 10:18:28 2025

@author: Frederick
"""

import flopy 
import numpy as np
import pandas as pd
import geopandas as gpd
import time
import os
import matplotlib.pyplot as plt

from flopy.utils.gridintersect import GridIntersect
from shapely.geometry import Point
#%%
sim_ws = "./SteadyStateModel/"
sim_name = "mfsim.nam"
sim = flopy.mf6.MFSimulation.load(sim_ws=sim_ws,sim_name=sim_name)
gwf = sim.get_model()

if not os.path.exists(sim_ws+"GW40_baseline.hds"):
    sim.run_simulation()
    hds0 = gwf.output.head().get_data()
    current_file_name = sim_ws+"GW40.hds"
    new_file_name = sim_ws+"GW40_baseline.hds"
    time.sleep(1)
    os.rename(current_file_name, new_file_name)
else:
    hds0 = flopy.utils.HeadFile(sim_ws+"GW40_baseline.hds").get_data()
vgrid = gwf.modelgrid

vobs = pd.read_csv("./spatial/vobs.csv")
clobs = pd.read_csv("./spatial/vobs_tr_adaption.txt")

order = ['Kiebingen1', 'Kiebingen2', 'Kiebingen3','Kiebingen4','Kiebingen5','Kiebingen6','Altingen3','Breitenholz', 'Entringen1', 'Entringen2', 'Poltringen1', 'Poltringen2']
vobs["name"] = pd.Categorical(vobs["name"], categories= order, ordered = True)
vobs = vobs.sort_values("name").reset_index(drop=True)
clobs["name"] = pd.Categorical(clobs["name"], categories= order, ordered = True)
clobs = clobs.sort_values("name").reset_index(drop=True)
vobs.to_csv("./spatial/vobs_sorted.csv")
clobs.to_csv("./spatial/vobs_close_sorted.csv")

w = gwf.wel.stress_period_data.get_data()
cc = vgrid.xyzcellcenters
#%%

mapping = {
    "Kiebingen1": "TB Kiebingen 1",
    "Kiebingen2": "TB Kiebingen 2",
    "Kiebingen3": "TB Kiebingen 3",
    "Kiebingen4": "TB Kiebingen 4",
    "Kiebingen5": "TB Kiebingen 5",
    "Kiebingen6": "TB Kiebingen 6",
    "Altingen3": "TB Altingen 3",
    "Breitenholz": "TB Breitenholz",
    "Entringen1": "TB Entringen 1",
    "Entringen2": "TB Entringen 2",
    "Poltringen1": "TB Poltringen 1",
    "Poltringen2": "TB Poltringen 2"
}

#%%
fig, axs = plt.subplots(3, 4, figsize=(20, 12))
axs = axs.flatten()

# Font sizes
title_fontsize = 10
tick_fontsize = 8
colorbar_fontsize = 14
ax_idx = 0
welspd = w[0]
# vmin, vmax = 0,20

hq = np.zeros((12,12))
hqcl = np.zeros((12,12))
hq_sc = np.zeros((12,12))
hqcl_sc = np.zeros((12,12))
names = []
locs = {}

sens= pd.DataFrame(np.zeros((vobs.shape[0],vobs.shape[0])))
sens.index = [mapping[n].lower() for n in vobs.name]
sens.columns = [mapping[n].lower() for n in vobs.name]

for i in range(welspd.shape[0]):
    name = welspd[i][-1]
    if "tb" in name.lower() and not "altingen 1" in name.lower() and not "altingen 2" in name.lower():
        if not os.path.isfile(sim_ws+f"GW40_{name.lower()}.hds"):
            l,c = welspd[i][0]
            welspd[i][1]*=0.8  #reduce rate
            gwf.wel.stress_period_data.set_data({0:welspd})
            sim.write_simulation()
            sim.run_simulation()
            welspd[i][1]/=0.8 # increase rate to former level
            # time.sleep(10)
            hds = gwf.output.head().get_data()
            current_file_name = sim_ws+"GW40.hds"
            new_file_name = sim_ws+f"GW40_{name.lower()}.hds"
            time.sleep(1)
            os.rename(current_file_name, new_file_name)
        else:
            hds = flopy.utils.HeadFile(sim_ws+f"GW40_{name.lower()}.hds").get_data()
        
        l,c = welspd[i][0]        
        dq = welspd[i][1]*0.2
        names.append(name)
        
        x,y = cc[0][c], cc[1][c]
        locs[name] = [x,y]
        diff = hds[l]-hds0[l]
        print(name)
        print(np.sqrt(diff**2).max())
        vmin,vmax = 0, np.sqrt(diff**2).max()
        
        for v in range(vobs.shape[0]):
            lay,cel = vobs.loc[v,"layer"], vobs.loc[v,"cell"]
            h0 = hds0[lay,0,cel]
            h1 = hds[lay,0,cel]
            hq[v,ax_idx] = round(h1-h0,4)
            hq_sc[v,ax_idx] = round((h1-h0)/abs(dq),8)
            lay,cel = clobs.loc[v,"layer"], clobs.loc[v,"cell"]
            h0 = hds0[lay,:,cel][0]
            h1 = hds[lay,:,cel][0]
            hqcl[v,ax_idx] = round(h1-h0,4)
            hqcl_sc[v,ax_idx] = round((h1-h0)/abs(dq),8)
            sens.loc[name, mapping[vobs.loc[v,"name"]].lower()] = round((h1-h0)/abs(dq),8)
        ax = axs[ax_idx]
        ax_idx+=1
        
        pmv = flopy.plot.PlotMapView(gwf, ax=ax)
        im = pmv.plot_array(
            diff,
            cmap="jet",
            alpha=0.5,
            masked_values=[np.log10(1)],
            vmin=vmin,
            vmax=vmax
        )
        ax.set_xlim(x-50,x+50)
        ax.set_ylim(y-50,y+50)
        # c = pmv.contour_array(diff[i], levels = np.arange(vmin,vmax,20), cmap = "Grays")  #colors = "lightgrey"
        # plt.clabel(c, fmt="%.1f", fontsize=7)
        ax.set_title(
            f"{name}, max diff: {round(vmax,2)}, dQ: {round(dq,2)}",
            fontsize=title_fontsize
        )
    
        #ax.scatter(pp.x[pp.zone == z[i]], pp.y[pp.zone == z[i]], zorder=1, s=1)
        ax.set_aspect('equal')
        
# cbar_ax = fig.add_axes([0.2, -0.05, 0.6, 0.02])  
# cbar = fig.colorbar(im, ax=cbar_ax, orientation="horizontal", fraction=0.05, pad=0.1)
plt.tight_layout()
plt.savefig("./img/wellheaddiffs.png")
plt.show()




#%%
vobs_xy = []
clobs_xy = []
for i in range(vobs.shape[0]):
    l,c = vobs.loc[i,"layer"],vobs.loc[i,"cell"]
    x,y = cc[0][c], cc[1][c]
    vobs_xy.append([x,y])
    l,c = clobs.loc[i,"layer"],clobs.loc[i,"cell"]
    x,y = cc[0][c], cc[1][c]
    clobs_xy.append([x,y])

vobs["x"] = np.array(vobs_xy)[:,0]
vobs["y"] = np.array(vobs_xy)[:,1]
clobs["x"] = np.array(clobs_xy)[:,0]
clobs["y"] = np.array(clobs_xy)[:,1]

vobs.to_csv("./spatial/vobs_xy.csv")
clobs.to_csv("./spatial/clobs_xy.csv")
#%%

pd.DataFrame(hq, columns = names, index = names).to_csv("./WellData/hq.csv")
pd.DataFrame(hq_sc, columns = names, index = names).to_csv("./WellData/hq_sc.csv")
pd.DataFrame(hqcl, columns = names, index = names).to_csv("./WellData/hqcl.csv")
pd.DataFrame(hqcl_sc, columns = names, index = names).to_csv("./WellData/hqcl_sc.csv")

#%%
dx = 1000
fig, axs = plt.subplots(3, 4, figsize=(20, 12))
axs = axs.flatten()
title_fontsize = 14
tick_fontsize = 12
colorbar_fontsize = 14
ax_idx = 0
welspd = w[0]
hds0 = flopy.utils.HeadFile(sim_ws+f"GW40_baseline.hds").get_data()
vmin, vmax = -2,2
for i in range(welspd.shape[0]):
    name = welspd[i][-1]
    if "tb" in name and not "altingen 1" in name and not "altingen 2" in name:
        l,c = welspd[i][0]
        hds = flopy.utils.HeadFile(sim_ws+f"GW40_{name}.hds").get_data()
        ax = axs[ax_idx]
        ax_idx+=1
        
        x,y = cc[0][c], cc[1][c]
        diff = hds[l]-hds0[l]
        print(name)
        print(np.sqrt(diff**2).max())
        
        vmin,vmax = 0, np.sqrt(diff**2).max()
        pmv = flopy.plot.PlotMapView(gwf, ax=ax)
        im = pmv.plot_array(
            hds[l]-hds0[l],
            cmap="jet",
            alpha=0.5,
            masked_values=[np.log10(1)],
            vmin=vmin,
            vmax=vmax
        )
        
        ax.set_xlim(x-dx,x+dx)
        ax.set_ylim(y-dx,y+dx)
        # c = pmv.contour_array(hds[i], levels = np.arange(vmin,vmax,20), cmap = "Grays")  #colors = "lightgrey"
        # plt.clabel(c, fmt="%.1f", fontsize=7)
        ax.set_title(
            f"{name}",
            fontsize=title_fontsize
        )

        #ax.scatter(pp.x[pp.zone == z[i]], pp.y[pp.zone == z[i]], zorder=1, s=1)
        ax.set_aspect('equal')
        
# cbar_ax = fig.add_axes([0.2, -0.05, 0.6, 0.02])  
# cbar = fig.colorbar(im, cax=cbar_ax, orientation="horizontal", fraction=0.05, pad=0.1)
plt.tight_layout()
plt.show()