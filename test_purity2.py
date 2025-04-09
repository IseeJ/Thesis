import numpy as np
import matplotlib.pyplot as plt
from lmfit.models import ExpressionModel

import Purity as pur
import RigolTools as rigol

from scipy.signal import savgol_filter





# apply the model to data...
runs = range(42, 56)
fnames = []
for run in runs:
    fnames.append(f'data/3rdLar{run}.csv')
vcath = [280, 250, 220, 180, 150, 120, 90, 90, 60, 50, 40, 30, 20, 10]
vcath = [-v for v in vcath]

dfs = []
for ii, fname in enumerate(fnames):
    print(f"ii, fname = {ii}, {fname}")
    dfs.append(rigol.read_csv(fname, ch3='anode', ch4='cathode', tunit='us', vunit='mV'))
    dfs[-1].attrs['vcath'] = vcath[ii]
    
## remove mean of baseline
for df in dfs:
    rigol.subtract_baseline(df, chans=['cathode', 'anode'])

"""
# just do 3 waveforms for now
for ii in range(3):#len(dfs)):
    td = 15.0 # us
    Cf = 1.4
    vgain = 2.0
    # FIXME: model assumes peak starts at first entry, but not true for data!
    Q0 = np.max(np.abs(dfs[ii]['cathode'].values))*Cf*td/vgain
    result = vmod.fit(dfs[ii]['cathode'], x=dfs[ii]['time'], Q0=Q0, td=td)
    print(result.fit_report())
    plt.plot(dfs[ii]['time'], dfs[ii]['cathode'], linewidth=0.5, color='black')
    plt.plot(dfs[ii]['time'], result.best_fit, linewidth=0.5, color='red', linestyle='--')
    plt.savefig(f'junk_{ii}.pdf')
    plt.clf()
"""

Q0_list = []
E1_list = []
Vmax_list = []
td_list = []
Qerr_list =[]
F_list = []

#smooth then trim data
for ii in range(14):  
#    td = 15.0  # us
#    Cf = 1.4#

    vgain = 2.0

    # find start of the signal
    # need a better way to determine first point of falling edge (right now it's just when signal amp>0.5, won't work for all....)
    #signal_start = np.where(np.abs(dfs[ii]['cathode'].values)>1)[0][0]
    time = dfs[ii]['time'].values
    cathode = dfs[ii]['cathode'].values
    cathode_smooth = savgol_filter(cathode, window_length=20, polyorder=2)


    slope = np.diff(cathode_smooth)
    # turning point
    print('max slope = ',np.max(np.abs(slope)))
    base = -0.4*np.max(np.abs(slope))
    print('base = ', base)
    start = np.argmax(slope < base)
    peak = np.argmin(cathode_smooth)


    V_max = -np.min(cathode_smooth)
    t1 = time[start]
    t2 = time[peak]
    td = t2-t1
    
    #Q0 = np.max(np.abs(cathode_new)) * Cf * td / vgain
    #result = vmod.fit(cathode_new, x=time_new, Q0=Q0, td=td)
    #print(fnames[ii])
    #print(result.fit_report())

    CV = vcath[ii]
    E1 = CV/1.65

    Cf = 1.4 #pF
    Rf = 100 #MOhm
    tauf = Rf*Cf
    F = (1/Cf)*(tauf/td)*(1-np.exp(-td/tauf))
    
    Q0_val =(V_max/vgain)/F
    td_val = td


    Q_err1 = 1/F
    xx = np.exp(td/tauf)-1
    Q_err2 = (V_max*Cf/tauf)*(((tauf*xx-td)*np.exp(td/tauf))/(tauf*xx**2))
    Q_err = np.sqrt(Q_err1**2+Q_err2**2)
    E1_list.append(E1)
    Vmax_list.append(V_max)
    Q0_list.append(Q0_val)
    td_list.append(td_val)
    Qerr_list.append(Q_err)

    F_list.append(F)
    plt.figure(dpi=200)
    
    plt.axvline(time[start],  linewidth=0.5, color='b', linestyle='--', label='signal start')
    plt.axvline(time[peak],  linewidth=0.5, color='b', linestyle='--', label='signal peak')
    plt.plot(dfs[ii]['time'].values,cathode_smooth*1e3, linewidth=0.5, color='black', label = 'data')       
    #plt.plot(dfs[ii]['time'].values,  dfs[ii]['cathode'].values, linewidth=0.5, color='black')
    #plt.plot(time_new, cathode_new, linewidth=0.5, color='black')
    plt.title(fnames[ii])

    plt.xlabel("Time [µs]")
    plt.ylabel('Voltage [mV]')
    plt.legend()
    plt.savefig(f'data/new/{ii}.png')
    plt.clf()




import csv

csv_filename = "data/fit_results2.csv"
with open(csv_filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["CV","E1","Vmax", "Q0", "td","F"])
    for e,a,b,c,d,f,Q in zip(vcath,E1_list, Vmax_list, Q0_list, td_list, F_list, Qerr_list):
        writer.writerow([e,a,b,c,d,f,Q])

