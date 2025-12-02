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


def update_wel_from_dataframe2(template_wel_path, nwf, output_path):
    """
    Create a new MODFLOW 6 WEL file with one period per row in 'wellfile'.
    Keeps all entries from the original file, replacing only values where
    the well name matches a column in 'wellfile'.
    """
    output_lines = []
    with open(template_wel_path, "r") as f:
        template_lines = f.readlines()
    goon = False
    per = np.nan
    for line in template_lines:
        if "END period" in line:
            goon = False
            newline = line
        if goon:
            n = line.split("'")[-2]
            ls = line.split()
            l,c,r = ls[:3]
            newrate = round(-nwf[n].iloc[per-1],6)
            if newrate>0:
                newrate*=-1
            if newrate == 0:
                newrate = 0.0
              
            lint = len(str(int(newrate)))
            lf = len(str(newrate))
            ldec = len(str(round(newrate,8)))-(lint+1)
            newline = f"  {l} {c}" + (8-lint)*" " + str(round(newrate,8)) + (8-ldec)*"0" + f"  '{n}' \n"
        if "BEGIN period" in line:
            goon = True
            per = int(line.split()[-1])
            newline = line
        elif not goon:
            newline = line
        output_lines.append(newline)
    with open(output_path, "w") as f:
        f.writelines(output_lines)

def update_wel_from_dataframe(template_wel_path, wellfile, output_path):
    """
    Create a new MODFLOW 6 WEL file with one period per row in 'wellfile'.
    Keeps all entries from the original file, replacing only values where
    the well name matches a column in 'wellfile'.
    """
    with open(template_wel_path, "r") as f:
        template_lines = f.readlines()

    # Find the lines that define well entries (with a regex)
    wel_line_pattern = re.compile(
        r'^\s*(\d+)\s+(\d+)\s+([-+]?\d*\.?\d+(?:[Ee][-+]?\d+)?)\s+("?[\w\s\.-]+"?)'
    )

    # Identify which lines belong to each period
    period_blocks = []
    current_block = []
    for line in template_lines:
        if line.strip().startswith("BEGIN period"):
            current_block = [line]
        elif line.strip().startswith("END period"):
            current_block.append(line)
            period_blocks.append(current_block)
            current_block = []
        else:
            if current_block is not None:
                current_block.append(line)

    # Build output WEL file
    header = []
    # Everything before the first period is header
    for line in template_lines:
        if line.strip().startswith("BEGIN period"):
            break
        header.append(line)

    output_lines = []
    output_lines.extend(header)

    # Iterate over each period = each row in DataFrame
    for i, (date, row) in enumerate(wellfile.iterrows(), start=1):
        # Use template of first period as base
        template_period = period_blocks[0]
        new_period = []

        # Replace BEGIN line
        new_period.append(f"BEGIN period  {i}\n")

        for line in template_period[1:-1]:  # skip BEGIN/END
            m = wel_line_pattern.match(line)
            if m:
                layer, cell, old_val, name = m.groups()
                name_clean = name.strip().lower().replace('"', '')
                if name_clean in [c.lower() for c in wellfile.columns]:
                    q = row[[c for c in wellfile.columns if c.lower() == name_clean][0]]
                    line = f"  {layer} {cell} {-q: .8E} {name}\n"
                new_period.append(line)
            

        # Add END line
        new_period.append(f"END period  {i}\n\n")

        output_lines.extend(new_period)

    # Write new file
    with open(output_path, "w") as f:
        f.writelines(output_lines)

def update_wel_from_dataframe(template_wel_path, wellfile, output_path):
    """
    Create a new MODFLOW 6 WEL file with one period per row in 'wellfile'.
    Keeps all entries from the original file, replacing only values where
    the well name matches a column in 'wellfile'.
    """
    with open(template_wel_path, "r") as f:
        template_lines = f.readlines()

    # Find the lines that define well entries (with a regex)
    wel_line_pattern = re.compile(
        r'^\s*(\d+)\s+(\d+)\s+([-+]?\d*\.?\d+(?:[Ee][-+]?\d+)?)\s+("?[\w\s\.-]+"?)'
    )

    # Identify which lines belong to each period
    period_blocks = []
    current_block = []
    for line in template_lines:
        if line.strip().startswith("BEGIN period"):
            current_block = [line]
        elif line.strip().startswith("END period"):
            current_block.append(line)
            period_blocks.append(current_block)
            current_block = []
        else:
            if current_block is not None:
                current_block.append(line)

    # Build output WEL file
    header = []
    # Everything before the first period is header
    for line in template_lines:
        if line.strip().startswith("BEGIN period"):
            break
        header.append(line)

    output_lines = []
    output_lines.extend(header)

    # Iterate over each period = each row in DataFrame
    for i, (date, row) in enumerate(wellfile.iterrows(), start=1):
        # Use template of first period as base
        template_period = period_blocks[0]
        new_period = []

        # Replace BEGIN line
        new_period.append(f"BEGIN period  {i}\n")

        for line in template_period[1:-1]:  # skip BEGIN/END
            m = wel_line_pattern.match(line)
            if m:
                layer, cell, old_val, name = m.groups()
                name_clean = name.strip().lower().replace('"', '')
                if name_clean in [c.lower() for c in wellfile.columns]:
                    q = row[[c for c in wellfile.columns if c.lower() == name_clean][0]]
                    line = f"  {layer} {cell} {-q: .8E} {name}\n"
                new_period.append(line)
            

        # Add END line
        new_period.append(f"END period  {i}\n\n")

        output_lines.extend(new_period)

    # Write new file
    with open(output_path, "w") as f:
        f.writelines(output_lines)

def check_heads(start,end,vobs,thr,filepath,plotting = True):
    hdsfile = flopy.utils.binaryfile.HeadFile(filepath)
    times = hdsfile.get_times()
    try:
        thr.set_index("name", inplace = True)
    except KeyError:
        pass
    try:
        vobs.set_index("name", inplace = True)
    except KeyError:
        pass
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
    return obs, mobs, flags, diffs


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
    deltah = delta @(hq)
    newdiffs-=deltah
    return delta, deltah

def translateWel2Df(wel_file):
    records = []
    period = None

    with open(wel_file, "r") as f:
        for raw_line in f:
            line = raw_line.strip()
            # skip empty or pure-comment lines
            if not line or line.startswith("#"):
                continue

            # remove inline comments that start with '#' (simple approach)
            # note: this will also remove '#' that appear after a space inside a quoted name,
            # which is uncommon. If that's an issue, more complex parsing is needed.
            line_no_comment = re.sub(r'\s+#.*$', '', line)

            if line_no_comment.startswith("BEGIN period"):
                # expect format: BEGIN period <n>
                try:
                    period = int(line_no_comment.split()[2])
                except Exception:
                    period = None
            elif line_no_comment.startswith("END period"):
                period = None
            elif period is not None:
                parts = line_no_comment.split()
                # ensure there are at least 3 tokens (layer, cellid, rate)
                if len(parts) < 3:
                    continue

                try:
                    layer = int(parts[0])
                except ValueError:
                    # skip malformed line
                    continue
                try:
                    cellid = int(parts[1])
                except ValueError:
                    # skip malformed line
                    continue

                # rate might be written like -123.45 or 1e3 etc.
                try:
                    rate = float(parts[2])
                except ValueError:
                    # attempt to clean (commas etc.)
                    rate_str = parts[2].replace(",", "")
                    try:
                        rate = float(rate_str)
                    except Exception:
                        # give up on this line
                        continue

                # Try to find a quoted name (single or double quotes)
                m = re.search(r"""(['"])(.*?)\1""", line_no_comment)
                if m:
                    name = m.group(2).strip().lower()
                else:
                    # No quoted name: assume name is the tokens after the rate
                    if len(parts) > 3:
                        name = " ".join(parts[3:]).strip().lower()
                    else:
                        # fallback to last token
                        name = parts[-1].strip().lower()

                # record period, name, absolute rate
                records.append((period, name, abs(rate)))

    # build DataFrame
    df = pd.DataFrame(records, columns=["period", "well", "rate"])

    # pivot into wide format (period rows, well columns)
    if df.empty:
        df_wide = pd.DataFrame()
    else:
        df_wide = df.pivot(index="period", columns="well", values="rate").fillna(0)

        # optional: assign dates to stress periods (example: daily starting 2019-11-01)
        start_date = pd.to_datetime("2019-11-01")
        df_wide.index = pd.date_range(start=start_date, periods=df_wide.shape[0], freq="D")
        df_wide.index.name = "Time"

    return df_wide



def optimizePumpingIter(diffs,hq,wellfile,wr_ts,safety_factor = 10):
    nwf = wellfile.copy()
    zdiffs = diffs.copy()
    newdiffs = diffs.copy()
    zdiffs[zdiffs>0]=0
    dq = zdiffs.copy()
    dq[dq>-99999] = 0
    capacity = zdiffs.copy()
    capacity[capacity>-99999] = 0
    iteration = 0
    while (newdiffs.values<0).sum()>0:
        for col in zdiffs.columns:
            dq[col] = zdiffs[col]/(hq.loc[col,col]/safety_factor)
            capacity[col] = diffs[col]/(hq.loc[col,col]/safety_factor)
        
        nwf = wellfile+dq
        nwf = np.maximum(nwf,0)
        deficit = (nwf-wellfile).sum(axis = 1)
        capacity = np.minimum(wr_ts-wellfile, capacity)
        for date in diffs.index:    
            if deficit.loc[date]!=0:
                cap = capacity.loc[date].sort_values(ascending = False)
                wincr,qi = [], 0
                count = 0
                while qi < abs(deficit.loc[date]):
                    count+=1
                    wincr = cap.nlargest(count).index
                    qincr = cap.nlargest(count).values
                    qi = qincr.sum()
                if count == 1:
                    nwf.loc[date,wincr] += abs(deficit.loc[date])
                else:
                    fracs = qincr/qincr.sum()
                    for w in range(wincr.shape[0]):
                        nwf.loc[date,wincr[w]] += abs(deficit.loc[date]*fracs[w])
        demand = nwf.sum(axis = 1)
        deltaq = nwf-wellfile
        for date in diffs.index:
            newdiffs.loc[date] -= deltaq.loc[date].iloc[3:-1]@hq
        print(iteration)
        iteration+=1

    for col in nwf.columns:
        if not "tb" in col:
            nwf[col] = wellfile[col]
    return nwf, demand

def optimizePumping(diffs,hq,wellfile,wr_ts,wr,safety_factor = 10, buffer = 0):
    nwf = wellfile.copy()
    zdiffs = diffs.copy()
    zdiffs[zdiffs>0]=0
    zdiffs-=buffer
    dq = zdiffs.copy()
    dq[dq>-99999] = 0
    capacity = zdiffs.copy()
    capacity[capacity>-99999] = 0

    for col in zdiffs.columns:
        dq[col] = zdiffs[col]/(hq.loc[col,col]/safety_factor)
        capacity[col] = diffs[col]/(hq.loc[col,col]/safety_factor)
    
    nwf = wellfile+dq
    nwf = np.maximum(nwf,0)
    deficit = (nwf-wellfile).sum(axis = 1)
    capacity = np.minimum(wr_ts-wellfile, capacity)
    if any(capacity[capacity>0].sum(axis = 1)<deficit.abs()):
        idx = np.where(capacity[capacity>0].sum(axis = 1)<deficit.abs())
        didx = diffs.index[idx]
        for w in wr.index:
            if "Poltringen" in w or "Entringen" in w:
                name = w.split(" ")[0]
                wr_ts.loc[didx,"tb "+name.lower() + " 1"] = wr.loc[w,"Spitzenentnahme [m^3/Tag]"]/2
                wr_ts.loc[didx,"tb "+name.lower() + " 2"] = wr.loc[w,"Spitzenentnahme [m^3/Tag]"]/2
            else:
                name = "tb " + w.lower()
                wr_ts.loc[didx,name] = wr.loc[w,"Spitzenentnahme [m^3/Tag]"]
        capacity = np.minimum(wr_ts-wellfile, capacity)
    for date in diffs.index:    
        if deficit.loc[date]!=0:
            cap = capacity.loc[date].sort_values(ascending = False)
            wincr,qi = [], 0
            count = 0
            while qi < abs(deficit.loc[date]):
                count+=1
                wincr = cap.nlargest(count).index
                qincr = cap.nlargest(count).values
                qi = qincr.sum()
            if count == 1:
                nwf.loc[date,wincr] += abs(deficit.loc[date])
            else:
                fracs = qincr/qincr.sum()
                for w in range(wincr.shape[0]):
                    nwf.loc[date,wincr[w]] += abs(deficit.loc[date]*fracs[w])
    demand = nwf.sum(axis = 1)

    for col in nwf.columns:
        if not "tb" in col:
            nwf[col] = wellfile[col]
    return nwf, demand


#%%

# path = "./Transient_welSimOpt/"
path = "./TransientModel/"

start = "2019-11-01"
# end = "2019-11-30"
end = "2020-10-31"
dr = pd.date_range(start,end,)


s = "2019-11-18"
e = "2019-11-24"
drc = pd.date_range(s,e)

writeNewFiles = False # if true, new wel and nam file are written
evaluation = True
prior_sim = True
useWelFile = True
override_limits = True # Flag that indicates if legal limits can be exceeded, specify 0<"buffer"<1

buffer = 0.5

if evaluation:
    wel_file0 = path+"GW40_0.wel"
    wel_file1 = path+"GW40_opt.wel"
    fp1 = path+"GW40.hds"
    fp0 = path+"GW40_beforeOpt.hds"

# wel_file = "./Transient_welSimOpt/GW40_opt.wel"
# fp = "./Transient_welSimOpt/GW40.hds"

wel_file = path+"GW40_0.wel"
fp = path+"GW40_beforeOpt.hds"

nn = ['Kiebingen1', 'Kiebingen2', 'Kiebingen3', 'Kiebingen4', 'Kiebingen5',
       'Kiebingen6', 'Altingen3', 'Breitenholz', 'Entringen1', 'Entringen2',
       'Poltringen1', 'Poltringen2']
nn2= ['TB Kiebingen 1', 'TB Kiebingen 2', 'TB Kiebingen 3', 'TB Kiebingen 4',
       'TB Kiebingen 5', 'TB Kiebingen 6','TB Altingen 3', 'TB Breitenholz', 'TB Entringen 1',
       'TB Entringen 2', 'TB Poltringen 1', 'TB Poltringen 2']
nn = pd.DataFrame([nn,nn2]).transpose()
nn.set_index(0,inplace = True)
restrictions = {
                # "TB Altingen 3": {"rate": 0, "start": "02.11.2019", "end": "30.11.2019", "year": 2019},
                # "TB Breitenholz":{"rate": 0, "start": "02.11.2019", "end": "30.11.2019", "year": 2019},
                # "TB Kiebingen 5":{"rate": 0, "start": "02.11.2019", "end": "30.11.2019", "year": 2019},
                }

restrictions = {k.lower(): v for k, v in restrictions.items()}
wr = pd.read_csv("./Wasserlinke/Wasserrechte ASG ab 2024.csv")
wr.set_index("Brunnen ", inplace = True)
wr.drop("BWV",axis = 0, inplace = True)

wr_ts = pd.DataFrame()
for k in wr.index:
    wr_ts[k] = np.zeros(dr.shape[0])
wr_ts.set_index(dr, inplace = True)


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

if restrictions is not None:
    for key in restrictions.keys():
        if key != "BWV":
            s = pd.to_datetime(restrictions[key]["start"], dayfirst = True)
            e = pd.to_datetime(restrictions[key]["end"], dayfirst = True)
            cond = np.multiply(wr_ts.index>=s, wr_ts.index<=e)
            wr_ts.loc[cond,key] = restrictions[key]["rate"]
 
    
if prior_sim:
    vobs = pd.read_csv("./spatial/vobs_close_sorted.csv") 
    thr = pd.read_csv("./WellData/thresholds.csv")
    # Check headfile for virtual observers -> thresholds exceeded?
   
    hq = pd.read_csv("./WellData/hqcl_sc.csv", index_col="Unnamed: 0")
    # for col in hq.columns:
    #     for row in hq.index:
    #         if row == col:
    #             pass
    #         elif "Kiebingen" in col and "Kiebingen" in row:
    #             hq.loc[row,col] *= 0.5
    #         else:
    #             hq.loc[row,col] *= 0.1
    hq *= np.eye(len(hq))# Sensitivities
    hq[hq<1e-2]*=2
    if evaluation:
        obs0, mobs0,flags0,diffs0 = check_heads(start,end,vobs,thr,filepath = fp0,plotting = False)
        obs1, mobs1,flags1,diffs1 = check_heads(start,end,vobs,thr,filepath = fp1,plotting = False)
        wellfile0 = translateWel2Df(wel_file0)
        wellfile1 = translateWel2Df(wel_file1)

        
        import matplotlib.pyplot as plt
        fig,axs = plt.subplots(3,4, figsize = (18,9))
        for v,ax in zip(vobs.index,axs.flatten()):
            ax.plot(mobs0.index,mobs0[v], color = "black", label = "Before Opt")
            ax.plot(mobs1.index,mobs1[v], color = "green", label = "After Opt")
            for i, ((ts0, row0), (ts1, row1)) in enumerate(zip(flags0.iterrows(),flags1.iterrows())):
                if row0[v] == 1:
                    ax.axvspan(ts0, ts0 + pd.Timedelta(days=1), color="yellow", alpha=0.1)
                elif row0[v] == 2:
                    ax.axvspan(ts0, ts0 + pd.Timedelta(days=1), color="red", alpha=0.1)
                if row1[v] == 1:
                    ax.axvspan(ts1, ts1 + pd.Timedelta(days=1), color="darkgoldenrod", alpha=0.1)
                elif row1[v] == 2:
                    ax.axvspan(ts1, ts1 + pd.Timedelta(days=1), color="darkred", alpha=0.1)
            ax.set_title(v)
            ax.set_xlabel("Date")
            ax.set_ylabel("Head [mNN]")
            ax.set_xticks([flags0.index[i] for i in np.arange(0,len(flags0),14)])
        ax.legend()
        plt.tight_layout()
        plt.show()
            
        mobs0.columns = [nn.loc[col].values[0].lower() for col in mobs0.columns]
        mobs1.columns = [nn.loc[col].values[0].lower() for col in mobs1.columns]      
        
    obs, mobs,flags,diffs = check_heads(start,end,vobs,thr,filepath = fp)
    diffs0 = diffs.copy()
    # diffs[diffs<0.5] = 0
    # obs1,flags1,diffs1 = check_heads(start,end,vobs,thr,filepath = "./TransientModel/GW40_beforeOpt.hds")
    diffs.columns = hq.index
    
    wr_ts = pd.DataFrame().from_dict({k: wr_ts[k] for k in diffs.columns})
    
    olddiffs = diffs.copy()
    if useWelFile:
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
            iteration+=1
            out = optimize_pumping_diff(hq, diffs.loc[date], q_max = wr_ts.loc[date], buffer = buffer)
            diffs.loc[date] = out["diff_new"]
            for key in restrictions.keys():
                if pd.to_datetime(restrictions[key]["start"],dayfirst = True)<=date and pd.to_datetime(restrictions[key]["end"],dayfirst = True) >= date:
                    diffs.loc[date,key.lower()] = 0
            diffs.loc[date] = diffs.loc[date].where(diffs.loc[date].isin(diffs.loc[date].nlargest(1)),0)
            dq = out["res"].x
            dq = pd.DataFrame(dq, index = hq.index)
            for col in diffs.columns:
                if col in hq.index:
                    nwf.loc[date,col] -= dq.loc[col].values[0]
            
            deficit = wellfile.loc[date,dq.index].sum()-nwf.loc[date,dq.index].sum()
            deltaq, deltah = redistribute(nwf.loc[date], deficit, hq*1.2, diffs.loc[date])
            nwf.loc[date, deltaq.index] += deltaq
            diffs.loc[date, deltah.index] += deltah
            # if date in drc:
            #     print(date)
            #     print(dq)
            #     print(deltaq)
    nwf_asg = nwf.copy()
    for col in nwf_asg.columns:
        if not "TB" in col and not "tb" in col:
            nwf_asg.drop(col,axis = 1,inplace = True)
    demand = nwf_asg.sum(axis = 1)
    
    # nwf,demand = optimizePumping(diffs,hq,wellfile,wr_ts, wr, safety_factor=4) # Safety Factor: larger is saver
    
    alreadyIncr = []
    if override_limits: 
        rates, wr_ts_withRes, excess = check_rates(pd.DataFrame(demand), nwf, wr, wr_ts, restrictions) 
        while excess[0]:
            exc = excess[1]
            capacity = diffs/np.diag(hq)
            for i in range(exc.shape[0]):
                if exc[i]:
                    date = dr[i]
                    cap = capacity.loc[date]
                    wellincr = cap.nlargest(5).index
                    for i in range(wellincr.shape[0]):
                        if wellincr[i] not in list(restrictions.keys()) and wellincr[i] not in alreadyIncr:
                            wr_ts.loc[date,wellincr[i]] += cap[wellincr[i]]*buffer
                            alreadyIncr.append(wellincr[i])
                            break
            rates, wr_ts_withRes, excess = check_rates(demand, nwf, wr, wr_ts, restrictions) 
    else:
        rates, wr_ts_withRes, _ = check_rates(demand, nwf, wr, wr_ts, restrictions)
        
rates.to_csv("./WellRates/new_well_rates.csv")

if writeNewFiles:
    oldwelfile = "GW40_0.wel"
    newwelfile = "GW40_opt.wel"
    update_wel_from_dataframe2(
        template_wel_path=path+oldwelfile,
        nwf=rates,
        output_path=path+newwelfile
    )
    
    with open(path+"GW40.nam", "r") as nam:
        lines = nam.readlines()
        
    with open(path+"GW40.nam", "w") as nam:    
        for line in lines:
            if oldwelfile in line or oldwelfile.lower() in line:
                print(line)
                l = line.split()
                l[1] = newwelfile
                newline = "  "+l[0]+"  "+l[1]+"  "+l[2]+"  \n"
            else:
                newline = line
            nam.write(newline)
    print("New files written successfully.")
            