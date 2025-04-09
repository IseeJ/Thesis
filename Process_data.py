import csv
import numpy as np
import matplotlib.pyplot as plt
from lmfit.models import ExpressionModel

import Purity as pur
import RigolTools as rigol

from scipy.signal import savgol_filter




def Vmax_cathode(path, fname, title):
    df = rigol.read_csv_2(path+'/'+fname, ch3='anode', ch4='cathode', tunit='us', vunit='mV')
    cathode = df['cathode'].values
    time = df['time'].values
    
    cathode_smooth = savgol_filter(cathode, window_length=20, polyorder=2)
    slope = np.diff(cathode_smooth)
    base = -0.4*np.max(np.abs(slope))
    start = np.argmax(slope < base)
    peak = np.argmin(cathode_smooth)


    V_max = -np.min(cathode_smooth)
    t1 = time[start]
    t2 = time[peak]
    td = t2-t1

    vgain = 2
    Cf = 1.4 #pF
    Rf = 100 #MOhm
    tauf = Rf*Cf
    F = (1/Cf)*(tauf/td)*(1-np.exp(-td/tauf))

    Q0 =(V_max/vgain)/F
    td = td

    Q_err1 = 1/F
    xx = np.exp(td/tauf)-1
    Q_err2 = (V_max*Cf/tauf)*(((tauf*xx-td)*np.exp(td/tauf))/(tauf*xx**2))
    Q_err = np.sqrt(Q_err1**2+Q_err2**2)


    plt.axvline(time[start],  linewidth=0.5, color='b', linestyle='--', label='signal start')
    plt.axvline(time[peak],  linewidth=0.5, color='b', linestyle='--', label='signal peak')
    plt.plot(time,cathode_smooth*1e3, linewidth=0.5, color='black', label = 'data')
    
    plt.title(title)

    plt.xlabel("Time [µs]")
    plt.ylabel('Voltage [mV]')
    plt.legend()
    plt.savefig(f'{path}/Vmax/{fname}.png')
    plt.clf()

    
    return V_max, F, Q0, Q_err, t1, t2


#def Fit_cathode(fname):
 #   df = rigol.read_csv(fname, ch3='anode', ch4='cathode', tunit='us', vunit='mV')
  #  cathode = df['cathode'].values
   # time = df['time'].values

    




    
# process
folder = 'Amplitude_vs_field'
filelist = '4thrun_data - Amplitude_vs_field.csv'

fnames_list = []
E2_list = []
results = []

with open(filelist, mode='r', newline='') as file:
    reader = csv.DictReader(file)

    for row in reader:
        filename = row['Filename']
        CV = int(row['C (V)'])
        AG = int(row['AG (V)'])
        E2 = AG/5.98
        A = int(row['A (V)'])
        ratio = float(row['E Ratio'])

        fnames_list.append(filename)
        E2_list.append(E2)

for i in range(len(fnames_list)):
    V_max,F, Q0, Q_err, t1, t2 = Vmax_cathode(folder,fnames_list[i],'E2 = '+str(E2_list[i])+' V/cm')
    results.append([E2_list[i],-V_max, F, Q0, Q_err, t1, t2, t2-t1])

with open(folder+'/Vmax_results.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['E2 (V/cm)','V_max','F', 'Q0', 'Q_err', 't1', 't2', 'T1'])  # Header
    writer.writerows(results)



