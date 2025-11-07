# -*- coding: utf-8 -*-
"""
Created on Tue Sep 16 14:44:06 2025

@author: Frederick
"""
import pandas as pd
import numpy as np
import pickle
import flopy
import re
from scipy.optimize import linprog
from demand2well import PoEn, check_rates

def check_heads(start,end,vobs,thr):
    hdsfile = flopy.utils.binaryfile.HeadFile("./model/GW40.hds")
    times = hdsfile.get_times()

    thr.set_index("name", inplace = True)
    vobs.set_index("name", inplace = True)
    obs = pd.DataFrame(np.zeros((len(times),len(vobs))))
    mobs = pd.DataFrame(np.zeros((len(times),len(vobs))))
    dr = pd.date_range(start,end)
    obs.set_index(dr,inplace = True)
    obs.columns = vobs.index
    mobs.set_index(dr,inplace = True)
    mobs.columns = vobs.index
    flags = pd.DataFrame(np.zeros((len(times),len(vobs))))        
    flags.set_index(dr,inplace = True)
    flags.columns = vobs.index
    diffs = pd.DataFrame(np.zeros((len(times),len(vobs))))      
    diffs.set_index(dr,inplace = True)
    diffs.columns = vobs.index
    for i,t in enumerate(times):
        hds = hdsfile.get_data(kstpkper=(0,t-1))
        for v in vobs.index:
            t0 = thr.loc[v,"yellowflag"]
            t1 = thr.loc[v,"redflag"]
            l,c = vobs.loc[v,"layer"],vobs.loc[v,"cell"]
            h = hds[l,:,c]
            mobs.loc[dr[i], v] = h
            diffs.loc[dr[i],v] = h-t1
            if h < t1:
                flags.loc[dr[i],v] = 2
            elif h < t0:
                 flags.loc[dr[i],v] = 1
    
    plotting = True
    if plotting:
        import matplotlib.pyplot as plt
        fig,axs = plt.subplots(3,4, figsize = (18,9))
        for v,ax in zip(vobs.index,axs.flatten()):
            ax.plot(mobs.index,mobs[v])
            for i, (ts, row) in enumerate(flags.iterrows()):
                if row[v] == 1:
                    ax.axvspan(ts, ts + pd.Timedelta(days=1), color="yellow", alpha=0.3)
                elif row[v] == 2:
                    ax.axvspan(ts, ts + pd.Timedelta(days=1), color="red", alpha=0.3)
            ax.set_title(v)
        plt.tight_layout()
        plt.show()
    return obs, flags, diffs




def optimize_pumping_diff(
    hq,                # (n,n) sensitivity matrix
    d,                 # (n,) current differences (can be negative if below threshold)
    q0=None,           # (n,) current pumping (optional, only used if you want absolute q_new)
    q_max=None,        # (n,) optional max increments
    cost=None,         # (n,) optional cost per unit pumping
    buffer = 0
):
    """
    Optimize pumping increments Δq so that all differences >= 0,
    with optional upper bounds on Δq.

    Parameters
    ----------
    S : ndarray (n,n)
        Sensitivity matrix (Δh = S @ Δq).
    d0 : ndarray (n,)
        Current differences (we want d0 + S@Δq >= 0).
    q0 : ndarray (n,), optional
        Current pumping rates (negative = extraction).
        If given, returns q_new = q0 + Δq.
    q_max : ndarray (n,), optional
        Maximum allowed increment for each well.
    cost : ndarray (n,), optional
        Cost vector for pumping increments. Default = ones.
    """

    n = len(d)
    if cost is None:
        cost = np.ones(n)

    # constraints: d0 + S Δq >= 0  ->  S Δq >= -d0
    A_ub = -hq.values
    b_ub = (d-buffer).values  # careful: b_ub = d0, since constraint is -S Δq <= d0

    # bounds: 0 <= Δq <= q_max (if given)
    if q_max is None:
        bounds = [(0, None) for _ in range(n)]
    else:
        bounds = [(0, q_max[i]) for i in range(n)]

    res = linprog(c=cost, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")

    if res.success:
        dq = res.x
        diff_new = d + hq @ dq
        out = {
            "success": True,
            "dq": dq,
            "diff_new": diff_new,
            "res": res
        }
        if q0 is not None:
            out["q_new"] = d + dq
        return out
    else:
        return {
            "success": False,
            "message": res.message,
            "res": res
        }

def redistribute(nwf, deficit, hq, newdiffs):
    fracs = newdiffs/newdiffs.sum()
    delta = fracs*deficit
    deltah = delta @hq
    newdiffs-=deltah
    return delta, deltah

def translateWel2Df(wel_file):
    wel_file = "./model/GW40_0.wel"
    records = []
    period = None
    
    with open(wel_file, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("BEGIN period"):
                period = int(line.split()[2])
            elif line.startswith("END period"):
                period = None
            elif period is not None and line and not line.startswith("#"):
                parts = line.split()
                # parse
                layer = int(parts[0])
                cellid = int(parts[1])
                rate = float(parts[2])
                # well name may have spaces or quotes
                name_match = re.findall(r'"(.*?)"', line)
                if name_match:
                    name = name_match[0].lower()
                else:
                    # handle unquoted names like gehrnfeld
                    name = parts[-1].lower()
                records.append((period, name, abs(rate)))
    
    # build DataFrame
    df = pd.DataFrame(records, columns=["period", "well", "rate"])
    
    # pivot into wide format
    df_wide = df.pivot(index="period", columns="well", values="rate").fillna(0)
    
    # optional: assign dates to stress periods (example: daily starting 2019-11-01)
    start_date = pd.to_datetime("2019-11-01")
    df_wide.index = pd.date_range(start=start_date, periods=df_wide.shape[0], freq="D")
    df_wide.index.name = "Time"
    return df_wide


#%%

start = "2019-11-01"
end = "2020-10-31"
dr = pd.date_range(start,end,)
prior_sim = True
useWelFile = True
override_limits = True # Flag that indicates if legal limits can be exceeded, specify 0<"buffer"<1
buffer = 0.2


nn = ['Kiebingen1', 'Kiebingen2', 'Kiebingen3', 'Kiebingen4', 'Kiebingen5',
       'Kiebingen6', 'Altingen3', 'Breitenholz', 'Entringen1', 'Entringen2',
       'Poltringen1', 'Poltringen2']
nn2= ['TB Kiebingen 1', 'TB Kiebingen 2', 'TB Kiebingen 3', 'TB Kiebingen 4',
       'TB Kiebingen 5', 'TB Kiebingen 6','TB Altingen 3', 'TB Breitenholz', 'TB Entringen 1',
       'TB Entringen 2', 'TB Poltringen 1', 'TB Poltringen 2']
nn = pd.DataFrame([nn,nn2]).transpose()
nn.set_index(0,inplace = True)
if prior_sim:
    vobs = pd.read_csv("./spatial/vobs_close_sorted.csv") 
    thr = pd.read_csv("./WellData/thresholds.csv")
    # Check headfile for virtual observers -> thresholds exceeded?
   
    hq = pd.read_csv("./WellData/hqcl_sc.csv", index_col="Unnamed: 0") # Sensitivities
    obs,flags,diffs = check_heads(start,end,vobs,thr)
    diffs.columns = hq.index
    if useWelFile:
        wel_file = "./model/GW40_0.wel"
        wellfile = translateWel2Df(wel_file)
        oldrates = wellfile.copy()
    else:
        wellfile = pd.read_csv("./WellRates/well_rates_19_20_orig.csv", parse_dates=True, index_col = "Time",dtype=float)
        wellfile.columns = [col.lower() for col in wellfile.columns]
    nwf = wellfile.copy()
    
    for date in dr:
        # No --> Bueno, do nothing
        # Yes --> check diference to threshold, compute delta Q from hq-Matrix, redistribute 
        iteration = 1
        while any(diffs.loc[date] < -1e-6):   # Optimization with tolerance of 1 Liter
            print(iteration)
            iteration+=1
            out = optimize_pumping_diff(hq, diffs.loc[date], buffer = 0)
            diffs.loc[date] = out["diff_new"]
            dq = out["res"].x
            dq = pd.DataFrame(dq, index = hq.index)
            for col in diffs.columns:
                if col in hq.index:
                    nwf.loc[date,col] -= dq.loc[col].values[0]
            deficit = wellfile.loc[date,dq.index].sum()-nwf.loc[date,dq.index].sum()
            deltaq, deltah = redistribute(nwf.loc[date], deficit, hq, diffs.loc[date])
            nwf.loc[date, deltaq.index] += deltaq
            diffs.loc[date, deltah.index] += deltah
    
    demand = nwf.sum(axis = 1)
    wr = pd.read_csv("./Wasserlinke/Wasserrechte ASG ab 2024.csv")
    wr.set_index("Brunnen ", inplace = True)
    wr.drop("BWV",axis = 0, inplace = True)
    
    wr_ts = pd.DataFrame()
    for k in wr.index:
        wr_ts[k] = np.zeros(demand.shape[0])
    wr_ts.set_index(demand.index, inplace = True)
    
    
    for k in wr.index:
        for d in wr_ts.index:
            wr_ts.loc[d,k] = wr.loc[k,"[m^3/Tag]"]

    wr_ts['TB Entringen 1'] = wr_ts["Entringen 1 u 2"]/2
    wr_ts['TB Poltringen 1'] = wr_ts["Poltringen 1 u 2"]/2
    wr_ts['TB Entringen 2'] = wr_ts["Entringen 1 u 2"]/2
    wr_ts['TB Poltringen 2'] = wr_ts["Poltringen 1 u 2"]/2
    wr_ts.drop(["Entringen 1 u 2", "Poltringen 1 u 2"], axis = 1, inplace = True)
    newcols = ['TB Altingen 3', 'TB Breitenholz', 'TB Kiebingen 1', 'TB Kiebingen 2', 'TB Kiebingen 3',
                     'TB Kiebingen 4', 'TB Kiebingen 5', 'TB Kiebingen 6']
    newcols.extend(wr_ts.columns[-4:])
    wr_ts.columns = [s.lower() for s in newcols]  
    restrictions = {}
    if override_limits: 
        excess = (True,None)
        while excess[0]:
            rates, wr_ts_withRes, excess = check_rates(demand, nwf, wr, wr_ts, restrictions)
            exc = excess[1]
            capacity = diffs/np.diag(hq)
            for i in range(exc.shape[0]):
                if exc[i]:
                    date = dr[i]
                    cap = capacity.loc[date]
                    wellincr = cap.idxmax()
                    wr_ts.loc[date,wellincr] *= (1+buffer)
                    
    else:
        rates, wr_ts_withRes, _ = check_rates(demand, nwf, wr, wr_ts, restrictions)
        
rates.to_csv("./WellRates/new_well_rates.csv")