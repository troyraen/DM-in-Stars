"""
Generates desc.csv from MESA LOGS files in plot_fncs.datadir.
Saves descDF.csv to plot_fncs.datadir.
Uses `history_pruned.data` files if `usepruned=True` (below) to avoid having to load the full files.

Most of this is adapted from ../../Glue/data_proc_README.py
"""

import sys
from pathlib import Path
import numpy as np
from collections import OrderedDict as OD
import mesa_reader as mr

import plot_fncs as pf
# Gpath = '/home/tjr63/DMS/mesaruns_analysis/Glue'
# if Gpath not in sys.path:
#     sys.path.append(Gpath)
import data_proc as dp # new data_proc.py is in this directory

usepruned = True

root = Path(pf.datadir)
# masses = np.around(np.arange(0.8, 5.1, 0.05).tolist(),decimals=2)
othmap={'basic':'0'}
list_Pcols = [ 'zone','logT','logRho','logL','logR','eta','q','eps_nuc','extra_heat' ]
list_Hcols = [ 'model_number','star_age','log_L', 'log_Teff','log_center_T','log_center_Rho',
# 'log_max_T',
'wimp_temp', 'mass_conv_core', 'center_h1', 'pp', 'cno']

# Get a dictionary of star models
# adapted from ../../Glue/data_proc.mkcbdir_flat() to cope with mesaruns directory structure
stars = {}
spin = 'Dep'
# step through the directory tree
for cbdir in root.iterdir():
    if not cbdir.is_dir(): continue
    cb = 'c' + cbdir.name[-1]
    # if int(cb[-1])>1: break

    for massdir in cbdir.iterdir():
        if not massdir.is_dir(): continue
        mass = float('.'.join(massdir.name.split('p')).strip('m'))
        # if mass<1.0 or cb not in ['c0','c1','c2','c3']: continue

        rootc = massdir / 'LOGS'
        hist = rootc / 'history_pruned.data' if usepruned else rootc / 'history.data'
        prof = rootc / 'profiles.index'

        if not hist.is_file(): continue # skip if history file doesn't exist
        print(f'Starting {massdir}')

        sid = len(stars.items())
        stars[sid] = OD([])
        sdic = stars[sid]
        dic = sdic

        # mass, spin, cboost, other descriptors
        dic['mass'] = mass
        dic['cb'] = cb
        dic['spin'] = spin
        dic['other'] = 'basic'
        desc = dp.get_desc(dic)

        dic['hist'] = mr.MesaData(str(hist))
        dic['pidx'] = mr.MesaProfileIndex(str(prof))
        # dic['l'] = mr.MesaLogDir(str(rootc))

        print(f'Loaded {massdir} to make descDF.csv')


stars1 = dp.sort_stars_dicts(stars)

dp.make_csv(stars1, othmap, list_Hcols,list_Pcols, clean=False, rt=str(root), write=['desc'])
