# library of functions useful in analysis of purity monitor data

import numpy as np
import pandas as pd

# functional form of a preamp output.
# assumes preamp transfer function is a simple exponential decay
# with timeconstant given by tau = Rf * Cf

# CR-110:
#   Rf=100 Mohm
#   Cf=1.4 pF 

# Units:
#   Time: microseconds
#   Capacitance:  picofarads
#   Resistance:   megaohms
#   Voltage: mV

def vout_preamp(tt, Cf=1.4, Rf=100, Q0=1.0, Vgain=None, td=5):
    # Cf [pF]: feedback capacitor (=1.4 for CR-110)
    # Rf [Mohm]: feedback resistor (=100 for CR-110)
    # Vgain [unitless]: voltage gain factor after the integrator (=2 for CR-110)
    # td [us]: drift time for an electron to cross the region
    #          e.g. time for an electron to go from cathode to cathode mesh

    # tau = Rf*Cf [us]
    tau = Rf*Cf # us
    alpha = 1e3*Q0/(td*Cf) # mV

    vout = alpha*tau*(1-np.exp(-tt/tau)) * (tt<td) # t<td
    vout += (alpha*tau)*(np.exp(td/tau)-1)*np.exp(-tt/tau)*(tt>=td) # t>td

    if Vgain:
        vout *= Vgain

    return vout
