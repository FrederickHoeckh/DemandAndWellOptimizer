# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 09:21:57 2025

@author: Frederick
"""
import pandas as pd
import numpy as np
import pickle
# import flopy
from scipy.optimize import linprog

def get_demand(PopVar, scenario, start, end, daily = True):
    """
    Parameters
    ----------
    PopVar : str
        Selects Variant of Population projection [V1-V5].
    scenario : str
        Selects how demand is estimated:
            "mean": Average demand is computed from historic records
            "max": Maximum demand is computed from historic records
            "detailed": Estimates demand based on historic data considering
                        temperature, day of week, holidays (for daily values)

    Returns
    -------
    Demand in m^3/d

    """
    if daily:    
        dr = pd.date_range(start,end, freq = "d")
    else:
        dr = pd.date_range(start,end, freq = "m")
    path = './'
    if dr[0].year <2021:
        ew = pd.read_csv(path + 'Population/ASG_ew_monthly.csv', parse_dates=True,index_col="Jahr")
    else:
        with open(path + 'Population/asg_ew_dic.pickle', 'rb') as handle:
            asg_ew_dic = pickle.load(handle)
        ew = asg_ew_dic[PopVar]
    if daily:
        ewm = ew.resample("d").mean().interpolate()
    else:
        ewm = ew.resample("m").mean().interpolate()
    if scenario == "mean" or scenario == "max":
        asg_wpc_weighted = pd.read_csv(path+'perHeadDemand/asg_wpc_WeightedMean.csv', parse_dates = True)
        demand = pd.DataFrame()
        demand["date"] = dr
        demand["demand"] = asg_wpc_weighted.loc[dr.month-1,scenario].values*ewm.loc[dr].sum(axis = 1).values
        demand.set_index("date", inplace = True)
    elif scenario == "detailed":
        asg_wpc = pd.read_csv(path+'perHeadDemand/asg_wpc.csv', parse_dates = True)
        asg_wpc.set_index(pd.to_datetime(asg_wpc["Unnamed: 0"]), inplace = True)
        asg_wpc.drop("Unnamed: 0", axis = 1, inplace = True)
        asg_wpc_max = asg_wpc.groupby(asg_wpc.index.month).max()
        asg_wpc_mean = asg_wpc.groupby(asg_wpc.index.month).mean()
        asg_wpc_std = asg_wpc.groupby(asg_wpc.index.month).std()
    else: 
        print(f"Scenario {scenario} not implemented, check for typos")
    
    return demand.div(demand.index.daysinmonth.T, axis = 0)

def PoEn(col):
    if "Entringen" in col or "Poltringen" in col:
        col2 = col.split(" ")[1].strip() + " 1 u 2"
    else:
        col2 = col.split("TB")[1].strip()
    return col2

def check_rates(demand_, rates_, wr, wr_ts_, restrictions, verbose = True, couple_compensation = False, daily = True):
    if verbose:
        print("Before optimizing")
        if any(wr_ts_.sum(axis=1).sub(demand_.T).T.values<0):
            idx = np.where(wr_ts_.sum(axis=1).sub(demand_.T).T<0)
            diff = wr_ts_.sum(axis=1).sub(demand_.T).T
            val = round(diff.min().iloc[0],2)
            print(f"Legal limits exceeded: {val} [$m^3/d$]")
        else:
            print("Everything within limits")
    
    years = rates_.index.year.unique()
    
    for yr in years:
        
        # flag = [entry for entry in restrictions if restrictions[entry]["year"] == yr]
        cond = np.where(rates_.index.year == yr)
        rates = rates_.iloc[cond]
        wr_ts = wr_ts_.iloc[cond]
        demand = demand_.iloc[cond]
        
        flag = pd.DataFrame()
        for col in wr_ts.columns:
            flag[col] = np.zeros(wr_ts.shape[0],dtype = int)
        flag.set_index(wr_ts.index, inplace = True)   
        if restrictions is not None:
            for key in restrictions.keys():
                if key != "BWV":
                    start = pd.to_datetime(restrictions[key]["start"], dayfirst = True)
                    end = pd.to_datetime(restrictions[key]["end"], dayfirst = True)
                    cond = np.multiply(wr_ts.index>=start, wr_ts.index<=end)
                    flag.loc[cond,key] = -1
        if couple_compensation:
            for entry in restrictions:
                if "Poltringen" in entry or "Entringen" in entry:
                    if int(entry[-1]) == 1:
                        incr = entry[:-1]+"2"
                    else:
                        incr = entry[:-1]+"1"
                    if incr in restrictions.keys():
                        pass
                    else:
                        wr_ts.loc[wr_ts[entry]==0,incr] *= 1.5  ## optional well couple compensation rate [0-1]: *= (1+comp)
        
        if any(demand.ge(wr_ts.sum(axis = 1),axis = 0)):
            rows = np.where(demand.ge(wr_ts.sum(axis = 1),axis = 0))[0]
            for row in rows:
                # if flag.index[row] == pd.to_datetime("31.07.2025", dayfirst = True):
                #     print("Lol")
                for col in wr_ts.columns:
                    if flag.loc[flag.index[row],col] >= 0:
                        tmp = pd.DataFrame()
                        tmp["s"] = np.zeros(flag.shape[1])
                        tmp.set_index(flag.columns,inplace = True)
                        for well in flag.columns:
                            tmp.loc[well,"s"] = flag.loc[flag[well]>=0,well].sum()#
                            if np.isnan(wr.loc[PoEn(well),"Spitzenentnahme [m^3/Tag]"]):
                                if daily:
                                    tmp.loc[well,"s"] = 61
                                else:
                                    tmp.loc[well,"s"] = 3
                        if daily:
                            tmp.loc[flag.loc[flag.index[row]]<0,"s"] = 61
                        else:
                            tmp.loc[flag.loc[flag.index[row]]<0,"s"] = 3
                        idx = (np.where(tmp["s"]==tmp["s"].min()))[0][0]
                        well = tmp.index[idx]
                        newrate = wr.loc[PoEn(well),"Spitzenentnahme [m^3/Tag]"]
                        if "Entringen" in well or "Poltringen" in well:
                            wr_ts.loc[wr_ts.index[row], well] = newrate/2
                        else:
                            wr_ts.loc[wr_ts.index[row], well] = newrate
                        # if well == "TB Altingen 3":
                        #     print("alt")
                        flag.loc[flag.index[row],well] = 1

                    frac_ts = wr_ts.div(wr_ts.sum(axis=1), axis = 0)
                    for wel in rates.columns:
                        rates.loc[:,wel] = (frac_ts.loc[:,wel]*demand.T).T.values
                    if not demand.iloc[row].ge(wr_ts.iloc[row].sum(),axis = 0).iloc[0]:
                        break
                      
        rates_[rates_.index.year == yr] = rates
        wr_ts_[wr_ts_.index.year == yr] = wr_ts
        
    if verbose:
        print("After optimizing")
        if any(wr_ts_.sum(axis=1).sub(demand_.T).T.values<0):
            idx = np.where(wr_ts_.sum(axis=1).sub(demand_.T).T<0)
            diff = wr_ts_.sum(axis=1).sub(demand_.T).T
            val = diff.min()
            print(f"Legal limits exceeded by up to {val} $m^3/d$")
            print(f"in at least one month in {yr}")
        else:
            print("Everything within limits")
    return rates_, wr_ts_


def demand2well(demand, restrictions):
    """
    

    Parameters
    ----------
    how : str
        how demand is distributed.
        rate: distribution by maximum possible rate
        day: by daily water rights
        annual: by annual water rights
    
    bwv: int/float
        
        if not None: fixed rate in L/s

    Returns
    -------
    None.

    """
    path = './'
    wr = pd.read_csv(path+"/Wasserlinke/Wasserrechte ASG ab 2024.csv")
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
    wr_ts.columns = newcols
    
    frac_ts_nores = wr_ts.div(wr_ts.sum(axis=1), axis = 0)
    
    if restrictions is not None:
        for key in restrictions.keys():
            if key != "BWV":
                start = pd.to_datetime(restrictions[key]["start"], dayfirst = True)
                end = pd.to_datetime(restrictions[key]["end"], dayfirst = True)
                cond = np.multiply(wr_ts.index>=start, wr_ts.index<=end)
                wr_ts.loc[cond,key] = restrictions[key]["rate"]
    
    
    frac_ts = wr_ts.div(wr_ts.sum(axis=1), axis = 0)
    diff = frac_ts_nores - frac_ts
    
    rates = pd.DataFrame()
    rates["date"] = demand.index
    rates.set_index("date", inplace = True)
    
    # rates_nores = rates.copy()
    # demand = demand.div(demand.index.daysinmonth, axis = 0)
    dmnd = demand.values.transpose().squeeze()
    for col in frac_ts.columns:
        rates[col] = frac_ts[col].values * dmnd
        # rates_nores[col] = frac_ts_nores[col].values * dmnd
    rates, wr_ts_withRes = check_rates(demand, rates, wr, wr_ts, restrictions)

    return rates,  wr_ts_withRes


def getWellRates(PopVar, scenario, start, end, restrictions, bwv = None,  unit = "m^3/d", file = None, pop = None):
    if file is None:
        demand = get_demand(PopVar, scenario, start, end)
    else:
        try:
            demand = pd.read_csv(file, index_col="date", parse_dates=True)
        except ValueError:
            demand = pd.read_csv(file, index_col=0, parse_dates=True)
        if type(pop) is not int:
            demand["demand"] = pop.values.squeeze()*demand.values.squeeze()*1.091
        else:
            demand["demand"] = pop*demand.values.squeeze()*1.091
        start = demand.index[0]
        end = demand.index[-1]
    # demand = wells_orig.iloc[:,:-4].sum(axis = 1)/1.09+0.09692*86400
    if bwv is not None:
        bwv_r = bwv
    else:
        bwv_r = 103
    bwv_d = bwv_r/1000*86400
    bwv_a = bwv_d*365
    bwv_ts = pd.DataFrame()
    bwv_ts["BWV"] = np.ones(demand.shape[0])*bwv_d
    bwv_ts.set_index(demand.index,inplace = True)
    
    if "BWV" in restrictions.keys():
        start = pd.to_datetime(restrictions["BWV"]["start"], dayfirst = True)
        end = pd.to_datetime(restrictions["BWV"]["end"], dayfirst = True)
        cond = np.multiply(bwv_ts.index>=start, bwv_ts.index<=end)
        bwv_ts.loc[cond,"BWV"] = restrictions["BWV"]["rate"]/1000*86400
    
    try:
        demand_after_bwv = demand-bwv_ts.values.squeeze()#.sub(bwv_ts, axis = 1)
    except ValueError:
        demand_after_bwv = demand-bwv_ts.values
    
    bwv_ts = pd.DataFrame(bwv_ts, index=demand.index)
    well_rates, wr_ts = demand2well(demand_after_bwv, restrictions)
    
    # full_dates = pd.date_range(start, end, freq="D")
    # well_rates = well_rates.reindex(full_dates).fillna(method="bfill")
    # bwv_ts     = bwv_ts.reindex(full_dates).fillna(method="bfill")
    # demand     = demand.reindex(full_dates).fillna(method="bfill")
    # wr_ts      = wr_ts.reindex(full_dates).fillna(method="bfill")
    return demand, demand_after_bwv, well_rates, bwv_ts, wr_ts


def plot_stuff(wells,bwv,demand,wr_ts,title = None, orig = None):
    res = pd.DataFrame()
    res["demand"] = demand
    res["wr"] = wr_ts.sum(axis = 1)
    res["rate"] = wells.sum(axis = 1)
    res["bwv"] = bwv
    import matplotlib.pyplot as plt
    fig,ax = plt.subplots()
    res.plot(ax=ax)
    if orig is not None:
        ax.plot(demand.index, orig)
    plt.title(title)
    plt.tight_layout()
    plt.show()
    


#%%
if __name__ == "__main__":
    PopVar = "V5"
    scenario = "mean"
    start = "2019-11-01"
    end = "2020-10-31"
    unit = "m^3/d"
    bwv = None
    dr = pd.date_range(start,end,)


    
    restrictions = {"TB Altingen 3": {"rate": 0, "start": "01.05.2025", "end": "30.07.2025", "year": 2025},
                    "TB Breitenholz":{"rate": 0, "start": "01.05.2025", "end": "30.07.2025", "year": 2025},
                    }
                    
    demand, demand_after_bwv, wells, bwv, wr_ts = getWellRates(PopVar, scenario, start, end, restrictions, bwv, unit = unit)
#%%             
    wellfile = pd.read_csv("./WellRates/wells_asg_swt_17_21.csv")
    wellfile["Time"]= pd.to_datetime(wellfile["Time"], format="%d.%m.%y")
    wellfile.set_index("Time",inplace = True)
    for col in wellfile.columns:
        if "Unnamed" in col:
            wellfile.drop(col, axis = 1, inplace = True)
    wells = wellfile.loc[pd.to_datetime(start):pd.to_datetime(end)]
    wells.to_csv("./well_rates_19_20_orig.csv")
    wells_orig = wells.loc[pd.to_datetime(start):pd.to_datetime(end)].copy()
    # plot_stuff(wells,bwv,wr_ts, "original")
#%%       
    bwv = None
    start = "2014-01-01"
    end = "2021-12-31"
    file = "./WaterDemand/syntheticDemand_14_21.csv"
    # restrictions = {"TB Breitenholz":{"rate": 0, "start": "01.03.2019", "end": "29.02.2020", "year": 2025}}
    restriction = {}
    ew = pd.read_csv("./Population/EinwohnerASG_corrected.csv", index_col = "Jahr")
    pop = ew.sum(axis = 1).loc[2014:2021]
    pop.index = pd.to_datetime(pop.index.astype(str), format="%Y")

    # Create daily date range
    date_range = pd.date_range(start="2014-01-01", end="2021-12-31", freq="D")

    # Reindex to daily and interpolate
    pop_daily = pop.reindex(date_range).interpolate(method="linear")

    # (Optional) set a name for the index
    pop_daily.index.name = "Date"
    pop = pop_daily.astype(int)
    
    demand, demand_after_bwv, wells, bwv_ts, wr_ts = getWellRates(PopVar, scenario, start, end, restrictions, bwv, unit = unit, file = file, pop = pop)
    # start = demand.index[0]
    # end = demand.index[-1]
    for col in wellfile.columns:
        if not "TB" in col:
            wells[col] = wellfile.loc[pd.to_datetime(start):pd.to_datetime(end, dayfirst=True),col]
    
    wells.to_csv("./WellRates/well_rates_14_21_syn_demand.csv")
    wells_K1 = wells.copy()
    demand_K1 = demand.copy()
    # plot_stuff(wells,bwv_ts,demand,wr_ts, "K1 reduced",wells_orig.sum(axis = 1))
#%%      
    bwv = None
    file = None
    start = "2014-01-01"
    end = "2021-12-31"
    restrictions = {"BWV": {"rate": 50, "start": "01.03.2019", "end": "29.02.2020", "year": 2025}}
    demand, demand_after_bwv, wells, bwv_ts, wr_ts = getWellRates(PopVar, scenario, start, end, restrictions, bwv, unit = unit, file = file, pop = 123000)
    # start = demand.index[0]
    # end = demand.index[-1]
    for col in wellfile.columns:
        if not "TB" in col:
            wells[col] = wellfile.loc[pd.to_datetime(start):pd.to_datetime(end, dayfirst=True),col]
    wells.to_csv("./WellRates/well_rates_19_synthetic_BWV.csv")
    wells_bwv = wells.copy()
    demand_bwv = demand.copy()
    # plot_stuff(wells,bwv_ts,demand,wr_ts, "BWV reduced",wells_orig.sum(axis = 1))

    