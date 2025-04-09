import sys
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import RigolTools as rigol
import numpy as np

runs = range(42, 56)
fnames = []
for run in runs:
    fnames.append(f'data/3rdLar{run}.csv')
vcath = [280, 250, 220, 180, 150, 120, 90, 90, 60, 50, 40, 30, 20, 10]
vcath = [-v for v in vcath]


#print(fnames)
#sys.exit()

dfs = []

for ii, fname in enumerate(fnames):
    print(f"ii, fname = {ii}, {fname}")
    dfs.append(rigol.read_csv(fname, ch3='anode', ch4='cathode', tunit='us', vunit='mV'))
    dfs[-1].attrs['vcath'] = vcath[ii]

## remove mean of baseline
for df in dfs:
    rigol.subtract_baseline(df, chans=['cathode', 'anode'])

plt.figure(dpi=200)
for df in dfs:
    print(np.min(df['cathode']))
    plt.plot(df['time'], df['cathode'], linewidth=0.5, label=df.attrs['vcath'])
    ##plt.plot(df['time'], df['anode'], rigol.acol)
plt.legend()
#plt.title(f"Cathode Signal vs. Cathode Field (with E2=E1)")
plt.xlabel("Time [us]")
plt.ylabel("Voltage [mV]")
plt.savefig('cathode_only_purity.pdf')
plt.savefig('cathode_only_purity.png')
plt.clf()

# smooth with Savitsky-Golay
wl = 20
po = 2
for df in dfs:
    df['cathode_filt'] = savgol_filter(df['cathode'].values, window_length=wl, polyorder=po)

plt.figure(dpi=200)
for df in dfs:
    plt.plot(df['time'], df['cathode_filt'], linewidth=0.5, label=df.attrs['vcath'])
#plt.title(f"Cathode: Smoothed Savitsky-Golay window={wl}, poly={po}")
plt.xlabel("Time [us]")
plt.ylabel("Voltage [mV]")
plt.legend()
plt.savefig('cathode_only_purity_filtered.pdf')
plt.savefig('cathode_only_purity_filtered.png')
plt.clf()

# Pick out the Vcath=-90V data (at least one of them... there are two)
i90 = vcath.index(-90) # this gives the first match -- that's fine.
print(i90)

# Fit function...
df = dfs[i90].copy()
plt.plot(df['time'], df['cathode'], linewidth=1, label=df.attrs['vcath'])
plt.axvline(0, linewidth=0.5, color='black')
plt.savefig('junk.pdf')
