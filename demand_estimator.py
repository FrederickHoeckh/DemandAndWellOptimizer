# -*- coding: utf-8 -*-
"""
Created on Wed Sep  3 08:53:37 2025

@author: Frederick
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 14:24:15 2024

@author: frede


# # # # # # # # # # TODO # # # # # # # # # # # # # # # # # 
Unsicherheiten richtig berechnen!!!
Überall Day of week benutzen!!

"""

import datetime
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pickle
import scipy
import geopandas as gpd
from joblib import Parallel, delayed
import scipy.stats as stats
from scipy.optimize import least_squares, minimize

#%% Plot
    
def isholiday(date, hol):
    yr = date.year
    holsub = hol[hol["Jahr"]==yr]
    k = 0
    for i in range(holsub.shape[0]):
        if date >= holsub["von"].iloc[i] and date <= holsub["bis"].iloc[i]:
            k = 1
            return True,holsub["Ferien"].iloc[i]
    if k == 0:
        return False,None
    

def get_rch4date(date, frac_mat, gwn, xg):
    if date is not None:
        if type(date) is str:
            date_str = " " + date + "-" + date + " "
        else:
            year = str(date)[:4]
            month = str(date)[5:7]
            day = str(date)[8:10]
            d = day +"."+ month +"."+ year
            date_str = " " + d + "-" + d + " "
        try:
            rch_vec = gwn[date_str]
        except KeyError:
            try:
                rch_vec = gwn[date_str[:-1]]
            except KeyError:
                rch_vec = gwn[date_str[1:-1]]
    else:
        rch_vec = gwn.iloc[:,2:-1].mean(axis=1)
    # print(rch_vec.sum())
    a = frac_mat*rch_vec
    
    return a.reshape(xg.shape)
    
    


#%% Temp Demand
def f(a,a2,x,y0,start, relative = False):       
    if type(x) is np.float64 or type(x) is int or type(x) is float:
        if x < start:
            out = y0+a*np.array(x-start)
        else:
            out = y0+a2*np.array(x-start)
    else:
        index = np.where(x == start)[0][0]
        out = np.concatenate((y0+a*np.array(x[:index]-start), y0+a2*np.array(x[index:]-start)), axis = None)
    if type(out) is float and out > 180:
        out = 180
    if relative:
        return out/y0
    else:
        return out/1000

def fit(x0):
    a,a2,y0,s = x0
    start = int(s)
    # fig, ax = plt.subplots(figsize = (10,8))
    # sc=ax.scatter(Tavg["Tm"].iloc[np.invert(condition_summerhol)], 
    #             perhead["mean"].iloc[np.invert(condition_summerhol)]*1000, s = 2, c = Tavg.index.dayofweek[np.invert(condition_summerhol)])
    # x = np.arange(-10,20)
    # # ax.plot(x,f(a,a2,x,y0,start), color = "red")
    # ax.set_ylim(60,180)
    # ax.set_xlim(-12,22)
    # plt.colorbar(sc)
    # plt.show()
    res = []
    for i in range(Tavg.shape[0]):
        res.append(perhead["mean"].iloc[i]*1000 - f(a,a2,Tavg["Tm"].iloc[i],y0,start))
    # print(np.mean(res))
    # return abs(np.mean(res))
    return res

#%%
def getDemand4date(date, wpc_dic, hol, scenario = "mid", T = None):
    """
    end of april (high), summer holidays (low), 
    mid june - end august (summerhigh)
    Scenario: high, mid, low
    
    1. Get time of year (< April, 20.04-20.05, Juni-Ende Juli, Sommerferien, nach Sommerferien)
    2. Hokidays
    3. Temperatureinfluss
    4. Einfluss DOW
    5. Unsicherheit
    
    """
    dow = date.dayofweek
    doy = date.dayofyear
    ishol, fer = isholiday(date, hol)
    
    if scenario == "mid":
        if doy >= wpc_dic["summer_doy"][0] and doy <= wpc_dic["summer_doy"][1]: #summer before holiday condition
            dmd = wpc_dic["avg_summer"]*wpc_dic["dow_avg_summer"][dow]/wpc_dic["dow_avg_summer"].mean()
        elif doy >= wpc_dic["garden_doy"][0] and doy <= wpc_dic["garden_doy"][1]: # Gardentime
            dmd = wpc_dic["avg_gardentime"]
        elif doy >= wpc_dic["xmas_doy"][0] and doy <= wpc_dic["xmas_doy"][1]: # Christmastime
            dmd = wpc_dic["avg_xmas"]
        else:
            dmd = wpc_dic["dow_avg"][dow]
        
        # 2. Check for Holiday
        if ishol:
            if fer == "Sommer":
                dmd = wpc_dic["avg_summerhol"]
            elif fer == "Winter":
                dmd = wpc_dic["avg_xmas"]
            else:
                dmd = wpc_dic["avg_hol"]
    elif scenario == "max":
        if doy >= wpc_dic["summer_doy"][0] and doy <= wpc_dic["summer_doy"][1]: #summer before holiday condition
            dmd = wpc_dic["max_summer"]*wpc_dic["dow_avg_summer"][dow]/wpc_dic["dow_avg_summer"].mean()
        elif doy >= wpc_dic["garden_doy"][0] and doy <= wpc_dic["garden_doy"][1]: # Gardentime
            dmd = wpc_dic["max_gardentime"]
        elif doy >= wpc_dic["xmas_doy"][0] and doy <= wpc_dic["xmas_doy"][1]: # Christmastime
            dmd = wpc_dic["max_xmas"]
        else:
            dmd = wpc_dic["dow_avg"][dow]+wpc_dic["dow_std"][dow]
        
        # 2. Check for Holiday
        if ishol:
            if fer == "Sommer":
                dmd = wpc_dic["max_summerhol"]
            elif fer == "Winter":
                dmd = wpc_dic["max_xmas"]
            else:
                dmd = wpc_dic["max_hol"]
    elif scenario == "mixed":
        a = np.random.random(1)[0]
        b = 1-a
        if doy >= wpc_dic["summer_doy"][0] and doy <= wpc_dic["summer_doy"][1]: #summer before holiday condition
            base = (a*wpc_dic["avg_summer"] + b*wpc_dic["max_summer"])
            dmd = base*wpc_dic["dow_avg_summer"][dow]/wpc_dic["dow_avg_summer"].mean()
        elif doy >= wpc_dic["garden_doy"][0] and doy <= wpc_dic["garden_doy"][1]: # Gardentime
            dmd = (a*wpc_dic["max_gardentime"]+b*wpc_dic["avg_gardentime"])
        elif doy >= wpc_dic["xmas_doy"][0] and doy <= wpc_dic["xmas_doy"][1]: # Christmastime
            dmd = (a*wpc_dic["max_xmas"]+b*wpc_dic["avg_xmas"])
        else:
            dmd = wpc_dic["dow_avg"][dow]+a*wpc_dic["dow_std"][dow]
        
        # 2. Check for Holiday
        if ishol:
            if fer == "Sommer":
                dmd = (a*wpc_dic["avg_summerhol"] + b*wpc_dic["max_summerhol"])
            elif fer == "Winter":
                dmd = (a*wpc_dic["avg_xmas"] + b*wpc_dic["max_xmas"])
            else:
                dmd = (a*wpc_dic["avg_hol"] + b*wpc_dic["max_hol"])
        
    if T is not None:
        temp,x0 = T
        a,a2,y0,s = x0
        dmd_fac = f(a,a2,temp,y0,s, relative = True)
        dmd *= dmd_fac
    # print(dmd)
    return dmd

def filter_by_date(df, start, end):
    return df[np.multiply(df.index.year>=start, df.index.year<=end)]

#%% Plot
def plot_doy_wpc_w_guess(perhead,hol,wpc_dic, x_fitted, plot_T = False, Tavg = None, scenario = "mid"):
    all_doym, all_doys = perhead["mean"],perhead["std"]
    fig, ax = plt.subplots(figsize=(20,6))
    numdays = 365
    # base = datetime.datetime.today()
    # date_list = [base - datetime.timedelta(days=x) for x in range(0, numdays)]
    # Set the locator
    # locator = mdates.MonthLocator()  # every month
    # # Specify the format - %b gives us Jan, Feb...
    # fmt = mdates.DateFormatter('%b')
    # font = {'family' : 'sans-serif',
    #         'weight' : 'normal',
    #         'size'   : 22}

    # matplotlib.rc('font', **font)
    ax.plot(all_doym.index, all_doym*1000, linewidth = 2, alpha = .4)
    dates = all_doym.index.values
    # for i in range(hol.shape[0]):
    #     ax.fill_betweenx(y = [0,150], x1 = hol["vdoy"][i], x2 = hol["bdoy"][i], color = "green", alpha = .07)
    ax.fill_between(all_doym.index,(all_doym+all_doys)*1000,(all_doym-all_doys)*1000, color = "grey", alpha = 0.4)
    d4d = pd.DataFrame()
    d4d["date"] = all_doym.index
    d4d["demand"] = np.zeros(all_doym.shape[0])
    d4d.set_index("date",inplace = True)
    if True:
        ax2 = ax.twinx()
        ax2.plot(all_doym.index, Tavg["Tm"], color = "red", linewidth = 3, alpha = .2)
        ax2.set_ylabel("Temperatur [°C]")
    for date in dates:
        temp = Tavg[Tavg.index==date]["Tm"].values[0]
        
        d4d["demand"][d4d.index==date] = getDemand4date(pd.to_datetime(date), wpc_dic, hol, scenario = scenario, T = [temp,x_fitted])
        # print(getDemand4date(date, wpc_dic, hol, scenario = "mid", T = [temp,x_fitted]))
    ax.scatter(d4d.index, d4d*1000, color = "black", s = 2)
    res = all_doym.values*1000-d4d.demand.values*1000
    var = np.var(res,ddof=1)
    start = pd.to_datetime("01.01.2021", dayfirst = True)
    end = pd.to_datetime("31.12.2021", dayfirst = True)
    d4d.loc[start:end].to_csv(f"./WaterDemand/synthetic_demand_{scenario}_19.csv")
    alpha = 0.05  # 95% confidence level
    n = len(all_doym)
    t_critical = stats.t.ppf(1 - alpha / 2, df=n-2)
    prediction_interval = t_critical * np.sqrt(var) * np.sqrt(1 + 1/n)
    
    # print("qdjkofsfha")
    # X = ax.xaxis
    # ax.set_xlabel("Day of Year", fontsize = 14)
    ax.set_ylabel(r"$\frac{l}{P \cdot d}$", fontsize = 24)
    # ax.set_ylim(95,150)
    ax.set_xlim(start,end)
    plt.grid()
    # X.set_major_locator(locator)
    # Specify formatter
    # X.set_major_formatter(fmt)
    plt.show()
    return d4d

    

#%% klimadaten
"""
Read climate data, Temperature is only relevant for demand forecast
WATCH OUT! Climate data is GWN-BW output data, here you can also use climate predictions
           Data MUST be in daily resolution
"""

sz = 500
grid_epsg   = 31467

frac_mat =scipy.sparse.load_npz("./spatial/rch_mapping"+str(sz)+"_full.npz")


outline     = gpd.read_file("./spatial/Model Outline/Outline_altered_east_gk3.shp").to_crs(grid_epsg)
minx, miny, maxx, maxy = outline.bounds.loc[0].round()

x = np.arange(minx, maxx, step = sz)
y = np.arange(miny, maxy, step = sz)
xg, yg = np.meshgrid(x, y)

start_date  = pd.to_datetime("01.01.2014", dayfirst=True) 
end_date    = pd.to_datetime("31.12.2021", dayfirst=True) 
date_range  = pd.date_range(start_date, end_date)

T_name      = "./temperature/Lurch01_Temperatur_Tageswert_ConstStep_1d_01111999.dat"
T           = pd.read_csv(T_name, delimiter = ",")

T_list = Parallel(n_jobs=8)(delayed(get_rch4date)(date, frac_mat, T, xg) for date in date_range)

Tavg = pd.DataFrame()
Tavg["Tm"] = [m.mean() for m in T_list]
Tavg["Ts"] = [m.std() for m in T_list]
Tavg.set_index(date_range, inplace =True)

#%% Wasserdemand

asg_munic = ["Altdorf", "Altenriet", "Ammerbuch", "Boeblingen", "Dettenhausen",
             "Hildrizhausen", "Holzgerlingen", "Rottenburg", "Schlaitdorf", 
             "Schoenaich", "Steinenbronn", "Tuebingen", "WalddorfHaeslach", "Waldenbuch", "Weil i. S." ]

ew = pd.read_csv("./Population/EinwohnerASG_corrected.csv", index_col = "Jahr")
hdh = np.zeros(ew.shape[0])
hdh[-1] = 3639
ew["Hildrizhausen"] = hdh
ew_init = ew.loc[2022].sum()

scaled_pred = pd.read_csv("./population/destatis_pred_scaled.csv",parse_dates=True, index_col="time")
ew_pred = ew_init*(1+scaled_pred)

perhead = pd.read_csv("./perHeadDemand/asg_wpc.csv", skiprows=[1])
perhead.set_index(pd.to_datetime(perhead["Unnamed: 0"]),inplace = True)
perhead.drop("Unnamed: 0", axis = 1,inplace = True)

perheaddoy = pd.DataFrame()
perheaddoy["mean"] = perhead["mean"].groupby(perhead.index.dayofyear).mean()
perheaddoy["std"] = perhead["mean"].groupby(perhead.index.dayofyear).std()

# Correction values
wpc_bb = 222.271
ew_share_bb = 0.1861

wpc_all = 150.1473

hol = pd.read_csv("./temporal/Holidays.csv", parse_dates=True)
hol["von"] = pd.to_datetime(hol["von"])
hol["bis"] = pd.to_datetime(hol["bis"])

# holiday condition
condition_hol = np.zeros(perhead.shape[0],dtype=float)
for i in range(hol.shape[0]):
    condition_hol += np.multiply(perhead.index<=pd.to_datetime(hol["von"].loc[i]), perhead.index>=pd.to_datetime(hol["bis"].loc[i]))
    
condition_hol.astype(bool)
perhead_nohol = perhead[np.invert(condition_hol.astype(bool))]
perhead_hol = perhead[condition_hol.astype(bool)]

summerdoy = [170,205]
endaprildoy = [107,121]
condition = np.add(perhead_nohol.index.dayofyear<summerdoy[0],perhead_nohol.index.dayofyear>summerdoy[1])

perhead_nohol_nosummer = perhead_nohol.iloc[condition]

condition_summer = np.add(perhead.index.dayofyear<summerdoy[0],perhead.index.dayofyear>summerdoy[1])
perhead_summer = perhead[np.invert(condition_summer)]

perhead_nohol.mean()
perhead_hol["mean"].mean()

# Summerholidays
summerhol = hol[hol["Ferien"] == "Sommer"]
condition_summerhol = np.zeros(perhead.shape[0])
for i in range(summerhol.shape[0]):
    condition_summerhol += np.multiply(perhead.index >= summerhol["von"].iloc[i],perhead.index <= summerhol["bis"].iloc[i]).astype(int)
condition_summerhol = condition_summerhol.astype(bool)
# Garden Time early to late april
gardendoy = [95,120]
condition_gt = np.multiply(perhead.index.dayofyear>gardendoy[0],perhead.index.dayofyear<gardendoy[1])
perhead_gt = perhead[condition_gt]

# Christmas Time 24.-30 Dec
xmasdoy = [359,365]
condition_xmas = np.multiply(perhead.index.dayofyear>xmasdoy[0],perhead.index.dayofyear<xmasdoy[1])
perhead_xmas = perhead[condition_xmas]

condition_normal_demand = np.multiply(np.invert(condition_gt), np.invert(condition_xmas))
condition_normal_demand = np.multiply(condition_normal_demand, np.invert(condition_hol.astype(bool)))
condition_normal_demand = np.multiply(condition_normal_demand, condition_summer)
# condition_normal_demand = np.multiply(condition_normal_demand, np.invert())


#%%
# Temperature Correction!!!
a = 0.3
a2 = 8
start = 10
y0 = 117
x0 = [a,a2,y0,start]

result = least_squares(fit,x0,loss = "cauchy")
x_fitted = result["x"]

#%% ConditionDict
"""
This is the most interesting part, below is the water-per-capita dictionary,
When the par capita demand is unusual at a certain time of the year it is categorized here
for example: holidays, gardening time etc.

The function "plot_doy_wpc_w_guess" produces timeseries (daily) of per capita water demand 
for a given time period, temperature, and scenario
here you can just plug in a future time period with according temperatures 
"""


wpc_dic = {
            "avg": perhead["mean"][condition_normal_demand].mean(),
            "avg_hol": perhead_hol["mean"].mean(),
            "avg_summer": perhead_summer["mean"].mean(),
            "avg_nosummer_nohol": perhead_nohol_nosummer["mean"].mean(),
            "avg_summerhol": perhead["mean"].iloc[condition_summerhol.astype(bool)].mean(),
            "avg_xmas": perhead["mean"].iloc[condition_xmas].mean(),
            "max": perhead["mean"][condition_normal_demand].max(),
            "max_hol": perhead_hol["mean"].max(),
            "max_summer": perhead_summer["mean"].max(),
            "max_nosummer_nohol": perhead_nohol_nosummer["mean"].max(),
            "max_summerhol": perhead["mean"].iloc[condition_summerhol.astype(bool)].max(),
            "max_xmas": perhead["mean"].iloc[condition_xmas].max(),
            "dow_avg": perhead["mean"][condition_normal_demand].groupby(perhead[condition_normal_demand].index.dayofweek).mean(),
            "dow_std": perhead["mean"][condition_normal_demand].groupby(perhead[condition_normal_demand].index.dayofweek).std(),
            "dow_avg_summer": perhead["mean"][condition_summer].groupby(perhead.index[condition_summer].dayofweek).mean(),
            "dow_avg_summerhol": perhead["mean"][condition_summerhol].groupby(perhead.index[condition_summerhol].dayofweek).mean(),
            "dow_avg_gardentime": perhead["mean"][condition_gt].groupby(perhead.index[condition_gt].dayofweek).mean(),
            "dow_max_summer": perhead["mean"][condition_summer].groupby(perhead.index[condition_summer].dayofweek).mean(),
            "dow_max_summerhol": perhead["mean"][condition_summerhol].groupby(perhead.index[condition_summerhol].dayofweek).mean(),
            "dow_max_gardentime": perhead["mean"][condition_gt].groupby(perhead.index[condition_gt].dayofweek).mean(),
            "avg_gardentime": perhead_gt["mean"].mean(),
            "max_gardentime": perhead_gt["mean"].mean(),
            "summer_doy": summerdoy,
            "garden_doy": gardendoy,
            "xmas_doy": xmasdoy,
           }

 
date = pd.to_datetime("12.05.2019")
temp = 15.1

getDemand4date(date, wpc_dic, hol, scenario = "max", T = [temp,x_fitted])         

# Correction for Böblingen
# * wpc_all/perhead["mean"].mean()*1e-3


wpc_mid = plot_doy_wpc_w_guess(perhead, hol, wpc_dic, x_fitted, plot_T = False, Tavg = Tavg, scenario = "mid")
wpc_max = plot_doy_wpc_w_guess(perhead, hol, wpc_dic, x_fitted, plot_T = False, Tavg = Tavg, scenario = "max")
wpc_mixed = plot_doy_wpc_w_guess(perhead, hol, wpc_dic, x_fitted, plot_T = False, Tavg = Tavg, scenario = "mixed")


# Apply Correction
wpc_mid *= wpc_all/perhead["mean"].mean()*1e-3

#%% Example calculation
start = "2014-01-01"
end = "2021-12-31"
dates = pd.date_range(start,end)
d4d19 = pd.DataFrame()
d4d19["demand"] = np.zeros(dates.shape[0])
d4d19.set_index(dates, inplace = True)
scenario = "mixed"
for date in dates:
    temp = Tavg[Tavg.index==date]["Tm"].values[0]
    dmnd = getDemand4date(pd.to_datetime(date), wpc_dic, hol, scenario = scenario, T = [temp,x_fitted])
    # Add correction to value and put in df
    d4d19["demand"][d4d19.index==date] = dmnd#*1.0352924526912013#*0.8978755419715864# *wpc_all/perhead["mean"].mean()*1e-3


d4d19.to_csv("./WaterDemand/syntheticDemand_14_21.csv")
