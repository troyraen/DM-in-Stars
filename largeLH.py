from collections import OrderedDict as OD
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from final_plots import plot_fncs as pf

dr = pf.mesaruns + '/RUNS_largeLH'
hpath = dr+ '/c6/m1p0/LOGS/history.data'

hdf = pd.read_csv(hpath, header=4, sep='\s+')

#-- plot luminosities v age
h = hdf.loc[hdf.star_age>1e7,:]
age = h.star_age
L = h.luminosity
LH = 10**h.log_LH
LHe = 10**h.log_LHe
Lnuc = 10**h.log_Lnuc
Lneu = 10**h.log_Lneu
Lgrav = h.eps_grav_integral
extra_L = h.extra_L
dic = OD([
        ('L', (L, ':')),
        ('LH', (LH, '-.')),
        ('LHe', (LHe, ':')),
        ('extra_L', (extra_L, '-')),
        ('Lneu', (Lneu, ':')),
        ('Lnuc', (Lnuc, '-')),
        ('Lgrav', (Lgrav, '-')),
      ])

# plot all luminosities
plt.figure()
for i, (sl, tup) in enumerate(dic.items()):
    l, ls = tup
    w = len(dic.keys())
    plt.plot(age,l,ls=ls,label=sl, zorder=w-i)
plt.semilogx()
plt.legend()
plt.xlabel('star_age')
plt.ylabel(r'luminosity [L$_\odot$]')
plt.tight_layout()
plt.show(block=False)

# plot excess
plt.figure()
plt.plot(age, L-Lnuc-Lgrav)
plt.semilogx()
plt.xlabel('star_age')
plt.ylabel(r'L-L$_{nuc}$-L$_{grav}$ [L$_\odot$]')
plt.tight_layout()
plt.show(block=False)
