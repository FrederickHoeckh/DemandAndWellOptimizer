# -*- coding: utf-8 -*-
"""
Created on Tue Sep 23 11:06:44 2025

@author: Frederick
"""

import flopy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score


#%%
def choose_threshold_from_scores(y_sim, y_meas_bool, metric="mismatch", return_all=False):
    """
    Choose new_threshold that best matches ymeas.

    Parameters
    ----------
    y_sim : pd.Series or 1d-array
        Simulated continuous values (e.g. vobsdf[n]).
    y_meas_bool : pd.Series or 1d-array of bool
        Measured boolean vector: True where measured < original_thrs (or however you defined).
    metric : str
        One of {"mismatch","false_negative","false_positive","f1","accuracy"}.
        Defaults to "mismatch".
    return_all : bool
        If True return (best_threshold, best_metrics_df).

    Returns
    -------
    best_threshold : float
    """
    # Convert to numpy arrays
    y = np.asarray(y_sim)
    ymeas = np.asarray(y_meas_bool).astype(bool)
    n = len(y)

    # Candidate thresholds: use unique y values, also include +/- inf edge cases
    uniq = np.unique(y)
    # To ensure thresholds that lead to all-True / all-False are considered, add extremes
    candidates = np.concatenate(([uniq[0] - 1e-9], uniq, [uniq[-1] + 1e-9]))

    results = []
    for t in candidates:
        ysim = (y < t)  # booleans
        # confusion components
        tp = np.sum( ymeas &  ysim)
        tn = np.sum(~ymeas & ~ysim)
        fp = np.sum(~ymeas &  ysim)
        fn = np.sum( ymeas & ~ysim)
        acc = (tp + tn) / n
        mismatches = fp + fn
        # F1: handle degenerate case
        if tp + fp == 0 or tp + fn == 0:
            f1 = 0.0
        else:
            f1 = f1_score(ymeas.astype(int), ysim.astype(int))
        results.append({
            "threshold": float(t),
            "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
            "mismatches": int(mismatches),
            "accuracy": float(acc),
            "f1": float(f1)
        })

    df = pd.DataFrame(results)

    # Choose best threshold according to metric
    if metric == "mismatch":
        best_row = df.loc[df["mismatches"].idxmin()]
    elif metric == "false_negative":
        best_row = df.loc[df["fn"].idxmin()]
    elif metric == "false_positive":
        best_row = df.loc[df["fp"].idxmin()]
    elif metric == "f1":
        best_row = df.loc[df["f1"].idxmax()]
    elif metric == "accuracy":
        best_row = df.loc[df["accuracy"].idxmax()]
    else:
        raise ValueError("Unknown metric")

    best_threshold = float(best_row["threshold"])
    if return_all:
        return best_threshold, df.sort_values(by=["mismatches", "threshold"]).reset_index(drop=True)
    return best_threshold


#%%
sim_ws = "./TransientModel/"
sim_name = "mfsim.nam"
sim = flopy.mf6.MFSimulation.load(sim_ws=sim_ws,sim_name=sim_name)
gwf = sim.get_model()
hdsfile = gwf.output.head()

times = hdsfile.get_times()
wel = gwf.wel.stress_period_data.get_data()

#%%
wd = {w[-1]: w[0] for w in wel[0]}

start,end = pd.to_datetime(["01.11.2019","31.10.2020"],dayfirst = True)
dr = pd.date_range(start,end)

#%%
# wn = pd.read_csv("well_names_ts.dat")
grabobs = pd.DataFrame()
wr = pd.read_csv("./WellRates/wells_asg_swt_17_21.csv", index_col = "Time")
wr.index = pd.to_datetime(wr.index, format = "%d.%m.%y")
wr = wr.loc[dr]
for col in wr.columns:
    if "Unnamed" in col:
        wr.drop(col,axis = 1, inplace = True)
    elif "TB" in col:
        grabobs[col] = wr[col]==0
wr.columns = [col.lower() for col in wr.columns]

#%%

obsdata = pd.read_csv("./obs/Grundwasserstände_1950_2025_wWells.csv", parse_dates = True, index_col="Datum").loc[dr]
# obsdata.columns = [col.lower() for col in obsdata.columns]
mobsdf = pd.DataFrame(np.ones(grabobs.shape)*np.nan)
mobsdf.columns = grabobs.columns
mobsdf.set_index(dr, inplace = True)
obsdf = mobsdf.copy()
vobsdf = mobsdf.copy()
for t in times:
    date = dr[int(t-1)]
    hds = hdsfile.get_data(kstpkper = (0,t-1))
    if any(grabobs.loc[date]):
        for i in range(grabobs.shape[1]):
            name = grabobs.columns[i]
            l,c = wd[name.lower()]
            mobs = hds[l,:,c][0]
            mobsdf.loc[date,name] = mobs
            if grabobs.loc[date,name]:
                obs = obsdata.loc[date,name]
                obsdf.loc[date,name] = obs
                
#%%
vobs = pd.read_csv("./spatial/vobs_close_sorted.csv")
vobs.drop("Unnamed: 0", axis = 1, inplace = True)

vobsdf.columns = [n for n in vobs.name]
vobsdf.set_index(dr, inplace = True)

for t,date in enumerate(dr):
    for j in range(vobs.shape[0]):
        hds = hdsfile.get_data(kstpkper = (0,t))
        n = vobs.loc[j,"name"]
        l,c = vobs.loc[j,"layer"],vobs.loc[j,"cell"]
        vobsdf.loc[date,n] = hds[l,:,c][0]       
        

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

original_thrs = pd.read_csv("./WellData/ActiveWellsMetaData.csv", index_col = "NAME")[:-3].TOPSCR
wsnt = pd.read_csv("../obs/WS_NT_wells.csv", parse_dates = True, index_col = "Datum")
wsat = pd.read_csv("../obs/WS_AT_wells.csv", parse_dates = True, index_col = "Datum")  

#%%
hows = ["mean","min","first"]
how = "Optimize" # hows[0]

names = []
thresholds = []

fig,axs = plt.subplots(3,4,figsize = (18,12), dpi = 300)
fig.suptitle(how)
for i, ax in enumerate(axs.flatten()):
    n = vobs.loc[i,"name"]
    name = mapping[n]
    names.append(n)
    print(name)
    if "Altingen" in n:
        obsname = n[:-1]
    elif "Poltringen" in n:
        obsname = "RWB_"+n
    else:
        obsname = n
    if obsname in wsat.columns:
        wsat[obsname].plot(ax=ax,color = "blue", label = "h measured", alpha = 0.7)
        if how == "mean":
            dh = (wsat[obsname]-vobsdf[n]).mean()
        elif how == "min":
            dh = wsat[obsname].min()-vobsdf[n].min()
        elif how == "first":
            for d in dr:
                if not np.isnan(wsat.loc[d,obsname]):
                    dh = wsat.loc[d,obsname]-vobsdf.loc[d,n]
        wellobs = wsat[obsname]
    else:
        wsnt[obsname].plot(ax=ax,color = "blue", label = "h measured", alpha = 0.7)
        if how == "mean":
            dh = (wsnt[obsname]-vobsdf[n]).mean()
        elif how == "min":
            dh = wsnt[obsname].min()-vobsdf[n].min()
        elif how == "first":
            for d in dr:
                if not np.isnan(wsnt.loc[d,obsname]):
                    dh = wsnt.loc[d,obsname]-vobsdf.loc[d,n]
        wellobs = wsnt[obsname]
    
    vobsdf[n].plot(ax=ax, label = "Sim h @ virtual obs.", color = "orange", alpha = 0.7)
    ax.hlines(original_thrs[name.lower()],0,1e6, color = "black", label = "Orig. Threshold", linestyle = "--")
    if how in hows:
        new_threshold = original_thrs[name.lower()]-dh
    else:
        ymeas = wellobs < original_thrs[name.lower()]
        new_threshold, metrics_df = choose_threshold_from_scores(vobsdf[n], ymeas, metric="mismatch", return_all=True)
        dh = new_threshold - original_thrs
    thresholds.append(new_threshold)
    
    ax.hlines(new_threshold,0,1e6, color = "green", label = "Sim. Threshold", linestyle = "--")
    x = vobsdf.index
    y = vobsdf[n]        
    ylim = ax.get_ylim()
    dy = ylim[1]-ylim[0]
    # Fill where below new threshold
    ax.fill_between(x,  ylim[0], ylim[0] + dy/2,
            where=(wellobs < original_thrs[name.lower()]),
            color="lightcoral", alpha=0.3, label="Below orig. thr.")     
    
    ax.fill_between(x, ylim[0] + dy/2, ylim[1],
            where=(y < new_threshold),
            color="lightgreen", alpha=0.3, label="Below sim. thr.")     
    print(round(original_thrs[name.lower()],2), round(new_threshold,2), round(dh[name.lower()],2))
    ax2 = ax.twinx()
    ax2.plot(wr[name.lower()], color = "grey", alpha = 0.3, label = "Q")
    ax.set_title(name)
    if i in [0,4,8]:
        ax.set_ylabel("h [mNN]")
    elif i in [3,7,11]:
        ax2.set_ylabel("Q [m$^3$/d]")
    if i in [8,9,10,11]:
        ax.set_xlabel("Date")
plt.tight_layout()
lines_1, labels_1 = ax.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
fig.legend(lines_1 + lines_2, labels_1 + labels_2, loc='lower center',
       bbox_to_anchor=(0.5, -0.02),   # slightly below the figure
       ncol=7)
plt.savefig(f"./img/{how}.png", bbox_inches='tight', dpi=300)
plt.show()

out = pd.DataFrame()
out["name"] = names
out["redflag"] = thresholds
out["yellowflag"] = thresholds
out["yellowflag"] += 0.2

out.to_csv("./WellData/thresholds.csv")
#%% 
fig,ax = plt.subplots(figsize = (9,6), dpi = 300)
n = "Poltringen2"
name = mapping[n]
print(name)
if "Altingen" in n:
    obsname = n[:-1]
elif "Poltringen" in n:
    obsname = "RWB_"+n
else:
    obsname = n
if obsname in wsat.columns:
    wsat[obsname].plot(ax=ax,color = "blue", label = "h measured", alpha = 0.7)
    if how == "mean":
        dh = (wsat[obsname]-vobsdf[n]).mean()
    elif how == "min":
        dh = wsat[obsname].min()-vobsdf[n].min()
    elif how == "first":
        for d in dr:
            if not np.isnan(wsat.loc[d,obsname]):
                dh = wsat.loc[d,obsname]-vobsdf.loc[d,n]
    wellobs = wsat[obsname]
else:
    wsnt[obsname].plot(ax=ax,color = "blue", label = "h measured", alpha = 0.7)
    if how == "mean":
        dh = (wsnt[obsname]-vobsdf[n]).mean()
    elif how == "min":
        dh = wsnt[obsname].min()-vobsdf[n].min()
    elif how == "first":
        for d in dr:
            if not np.isnan(wsnt.loc[d,obsname]):
                dh = wsnt.loc[d,obsname]-vobsdf.loc[d,n]
    wellobs = wsnt[obsname]

vobsdf[n].plot(ax=ax, label = "Sim h @ virtual obs.", color = "orange", alpha = 0.7)
ax.hlines(original_thrs[name.lower()],0,1e6, color = "black", label = "Orig. Threshold", linestyle = "--")
if how in hows:
    new_threshold = original_thrs[name.lower()]-dh
else:
    ymeas = wellobs < original_thrs[name.lower()]
    new_threshold, metrics_df = choose_threshold_from_scores(vobsdf[n], ymeas, metric="mismatch", return_all=True)
    dh = new_threshold - original_thrs
ax.hlines(new_threshold,0,1e6, color = "green", label = "Sim. Threshold", linestyle = "--")
x = vobsdf.index
y = vobsdf[n]        
ylim = ax.get_ylim()
dy = ylim[1]-ylim[0]
# Fill where below new threshold
ax.fill_between(x,  ylim[0], ylim[0] + dy/2,
        where=(wellobs < original_thrs[name.lower()]),
        color="lightcoral", alpha=0.3, label="Below orig. thr.")     

ax.fill_between(x, ylim[0] + dy/2, ylim[1],
        where=(y < new_threshold),
        color="lightgreen", alpha=0.3, label="Below sim. thr.")     
print(round(original_thrs[name.lower()],2), round(new_threshold,2), round(dh[name.lower()],2))
ax2 = ax.twinx()
ax2.plot(wr[name.lower()], color = "grey", alpha = 0.3, label = "Q")
ax.set_title(name)

ax.set_ylabel("h [mNN]")
ax2.set_ylabel("Q [m$^3$/d]")
ax.set_xlabel("Date")

# Combine both legends
handles1, labels1 = ax.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()

# Merge and remove duplicates
handles, labels = [], []
for h, l in zip(handles1 + handles2, labels1 + labels2):
    if l not in labels:
        handles.append(h)
        labels.append(l)

# Create single legend below the plot
ax.legend(handles, labels,
          loc='lower center',
          bbox_to_anchor=(0.5, -0.35),
          ncol=5,
          frameon=False)
plt.tight_layout()

plt.savefig(f"./img/Poltringen_2.png", bbox_inches='tight', dpi=300)
plt.show()
