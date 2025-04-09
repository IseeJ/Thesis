import numpy as np
import matplotlib.pyplot as plt
from lmfit.models import ExpressionModel

import Purity as pur
import RigolTools as rigol

from scipy.signal import savgol_filter

#fake data
tt = np.linspace(0, 300, num=1000)
Cf = 1.4 # pF
Rf = 100 # Mohm
Q0 = 1.0 # pC
td = 15.0 # us
Vgain = 2.0 # unitless: voltage gain factor after the integrator (=2 for CR-110)
vout = pur.vout_preamp(tt, Cf=Cf, Rf=Rf, Q0=Q0, td=td, Vgain=Vgain)

# add noise
vout += np.random.normal(size=tt.size, scale=20)

vout *= -1



# Parameters: tau, Cf, td, Q0
#vmod = ExpressionModel(" ((1e3*Q0*tau)/(td*Cf)) * ( (1-exp(-x/tau))*(x<td) + (exp(td/tau)-1)*(exp(-x/tau))*(x>=td) )")
#result = vmod.fit(vout, x=tt, Q0=Q0, tau=Rf*Cf, Cf=Cf, td=td)
"""
df = rigol.read_csv("data/20250312_2nd_LAr_run/lar1.csv", ch3='anode', ch4='cathode', tunit='us', vunit='mV')
print(df)
vout =df['cathode']
tt = df['time']
Q0 = 1.0 # pC
td = 15
"""

vmod = ExpressionModel(" (-2*(1e3*Q0*140)/(td*1.4)) * ( (1-exp(-x/140))*(x<td) + (exp(td/140)-1)*(exp(-x/140))*(x>=td) )")
result = vmod.fit(vout, x=tt, Q0=Q0, td=td)
print(result.fit_report())

plt.plot(tt, vout, 'k-', linewidth=0.5)
plt.plot(tt, result.best_fit, 'r--', linewidth=0.5)
#plt.axhline(vout.max()/np.exp(1), linestyle="--", linewidth=0.5)
#plt.axhline(1400, linestyle="--", linewidth=0.5, color='k')
plt.xlabel("Time [us]")
plt.ylabel("Voltage [mV]")
plt.savefig('junk.pdf')
plt.clf()

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
Q0_err_list = []
td_list = []
td_err_list = []


#smooth then trim data
for ii in range(14):  
#    td = 15.0  # us
    Cf = 1.4
    vgain = 2.0

    time = dfs[ii]['time'].values

    # find start of the signal
    # need a better way to determine first point of falling edge (right now it's just when signal amp>0.5, won't work for all....)
    #signal_start = np.where(np.abs(dfs[ii]['cathode'].values)>1)[0][0]
    cathode = dfs[ii]['cathode'].values*1e3
    cathode_smooth = savgol_filter(cathode, window_length=20, polyorder=2)


    slope = np.diff(cathode_smooth)
    # turning point
    print('max slope = ',np.max(np.abs(slope)))
    base = -0.4*np.max(np.abs(slope))
    print('base = ', base)
    signal_start = np.argmax(slope < base)
    """
    slope = np.diff(cathode_smooth)
    second_diff = np.abs(np.diff(slope))
    print(np.max(np.abs(second_diff)))
    signal_start = np.argmax(second_diff < 0)
    """
    signal_max = np.argmin(cathode_smooth)

    t1 = time[signal_start]
    t2 = time[signal_max]
    td = t2-t1
    
    time = dfs[ii]['time'].values
    time_new = dfs[ii]['time'].values[signal_start:]
    cathode_new = cathode_smooth[signal_start:]
    #cathode_new = dfs[ii]['cathode'].values[signal_start:]

    Q0 = np.max(np.abs(cathode_new)) * Cf * td / vgain
    result = vmod.fit(cathode_new, x=time_new, Q0=Q0, td=td)
    print(fnames[ii])
    print(result.fit_report())


    Q0_val = result.params['Q0'].value
    Q0_err = result.params['Q0'].stderr
    td_val = result.params['td'].value
    td_err = result.params['td'].stderr

    
    Q0_list.append(Q0_val)
    Q0_err_list.append(Q0_err)
    td_list.append(td_val)
    td_err_list.append(td_err)
    
    
    plt.axvline(time[signal_start],  linewidth=0.5, color='b', linestyle='--', label='signal start')
    plt.plot(dfs[ii]['time'].values,cathode_smooth, linewidth=0.5, color='black', label = 'data')       
    #plt.plot(dfs[ii]['time'].values,  dfs[ii]['cathode'].values, linewidth=0.5, color='black')
    #plt.plot(time_new, cathode_new, linewidth=0.5, color='black')
    plt.plot(time_new, result.best_fit, linewidth=0.5, color='red', linestyle='--', label='fit')
    plt.title(fnames[ii])

    plt.xlabel("Time [µs]")
    plt.ylabel('Voltage [mV]')
    plt.legend()
    plt.savefig(f'{fnames[ii]}.png')
    plt.clf()




import csv

csv_filename = "data/fit_results.csv"
with open(csv_filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Filename", "Q0", "Q0_err", "td", "td_err"])
    for vcath, q0, q0err, td, tderr in zip(vcath, Q0_list, Q0_err_list, td_list, td_err_list):
        writer.writerow([vcath, q0, q0err, td, tderr])

