import numpy as np
import mesa_reader as mr
import os
from pathlib import Path
#import collections as col
from collections import OrderedDict as OD
import pandas as pd
import importlib as imp # imp.reload(mod) to reload module
# from numpy import inf
import data_proc as dp

# ---------------------------------- #

root = '/Users/troyraen/Osiris/mesaruns/RUNS_2test_final/plotsdata/'
root = '/home/tjr63/Osiris/mesaruns/RUNS_2test_final/plotsdata/' # Korriban
# masses = [ 0.8 , 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]
masses = np.around(np.arange(0.8, 5.1, 0.05).tolist(),decimals=2)
# list_Pcols = [ 'zone','logT','logRho','logL','logR','net_nuclear_energy','net_energy','total_energy','eta','q','logE','eps_nuc','extra_heat','extra_L','log_extra_L','mass','dq','nx','np' ]#REG#
# list_Hcols = [ 'model_number','star_age','star_mass','log_dt','num_zones','log_LH','log_LHe','log_LZ','extra_L','log_extra_L','log_L','log_R','log_Teff','log_center_T','log_center_Rho','log_max_T','total_energy','total_extra_heating','total_energy_sources_and_sinks','wimp_temp','Nx_total','center_nx','center_np', 'extra_energy' ] #REG#
# list_Hcols = [ 'model_number','star_age','log_LH','log_LHe','log_LZ','extra_L','log_L','log_Lnuc','log_R','log_Teff','log_center_T','log_center_Rho','log_center_P','log_max_T','wimp_temp','Nx_total','center_nx','center_np', 'mass_conv_core', 'he_core_mass', 'center_h1', 'center_he4', 'total_mass_h1', 'pp', 'cno', 'tri_alfa' , 'star_mass', 'num_retries','num_backups','num_newton_iterations']
list_Pcols = [ 'zone','logT','logRho','logL','logR','eta','q','eps_nuc','extra_heat','extra_L','mass','dq','nx','np', 'mixing_type', 'h1' ]
list_Pcols = [ 'zone','logT','logRho','logL','logR','eta','q','eps_nuc','extra_heat' ]
list_Hcols = [ 'model_number','star_age','log_LH','log_LHe','log_LZ','extra_L','log_L','log_Lnuc','log_R', 'trace_mass_radius', 'log_Teff','log_center_T','log_center_Rho','log_center_P','log_max_T','wimp_temp','Nx_total','center_nx','center_np', 'mass_conv_core', 'he_core_mass', 'center_h1', 'center_he4', 'total_mass_h1', 'pp', 'cno', 'tri_alfa' , 'star_mass', 'center_eps_nuc', 'center_xheat']
list_Hcols = [ 'model_number','star_age','log_L', 'log_Teff','log_center_T','log_center_Rho','log_max_T','wimp_temp', 'mass_conv_core', 'center_h1', 'pp', 'cno']#, 'center_eps_nuc', 'center_xheat']

othmap={'basic':'0', \
        'inlist_premstowd':'1', \
        'isochrones_Aug':'2'}

# stars = dp.make_dicts(root, masses, load_prof=False)
stars = dp.make_dicts(root, [], load_prof=False) # 2nd arg = [] gets all available masses
# ---------------------------------- #
stars1 = dp.sort_stars_dicts(stars)
# dp.make_csv(stars1, othmap, list_Hcols,list_Pcols, clean='all', rt=root)
dp.make_csv(stars1, othmap, list_Hcols,list_Pcols, clean=False, rt=root+'/Glue', write=['desc'])

# Rewritting desc.csv to include mass_conv_core info for deltaTau plot
# masses without a c0 model need to be removed to create the description csv
mx = [2.03,2.08,2.13,2.13,2.18,2.18,2.23,2.23,2.28,2.28,2.33,2.33,2.38,2.38,2.43,\
        2.48,2.53,2.58,2.63,2.68,2.73,2.78,2.83,2.83,2.88,2.88,2.93,2.93,2.98,2.98,\
        3.03,3.03,3.08,3.08,3.13,3.18,3.18,3.23,3.23,3.28,3.28,3.33,3.33,3.38,3.38,\
        3.43,3.48,3.53,3.58,3.63,3.68,3.73,3.78,3.83,3.88,3.93,3.98,4.03,4.08,4.13,\
        4.18,4.23,4.28,4.33,4.38,4.43,4.48,4.53,4.58,4.63,4.68,4.73,4.78,4.83,4.88,4.93,4.98]
stars2 = { key: dic for key,dic in stars1.items() if dic['mass'] not in mx }
for k,s in stars2.items():
    try:
        del s['descDF']
    except:
        pass

# stmp = { key: dic for key,dic in stars1.items() if dic['mass'] in [1.0, 1.03] }
# dp.get_desc(stars1)
root = '/home/tjr63/tmp/'
dp.make_csv(stars2, othmap, list_Hcols,list_Pcols, clean='all', rt=root+'/Glue')

for key, dic in stars3.items():
    del dic['descDF']
dp.make_csv(stars3, othmap, list_Hcols,list_Pcols, clean=False, rt=root+'/Glue', write=['desc'])

# # Add column 'masscc_avg' to descDF:
# stars3 = stars2.copy()
# for key, dic in stars3.items():
#     h = dic['hist']
#     desc = dic['descDF']
#
#     # get df of history data
#     en = dp.findEV_enterMS(h) - 1 # get 1 model before MS for delta_age (below)
#     ex = dp.findEV_leaveMS(h)
#     df = pd.DataFrame(data={'star_age':h.star_age[en:ex],
#                             'mass_conv_core':h.mass_conv_core[en:ex]
#                             })
#     df['delta_age'] = df.star_age.diff()
#     df.drop(labels=0, axis=0, inplace=True) # drop the model before MS
#
#     MStau = h.star_age[ex] - h.star_age[en+1]
#     dic['descDF']['masscc_avg'] = (df.mass_conv_core* df.delta_age).sum() / MStau
#
# dp.make_csv(stars3, othmap, list_Hcols,list_Pcols, clean=False, rt=root, write=['desc'])
