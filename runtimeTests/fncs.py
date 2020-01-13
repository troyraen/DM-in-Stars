# paths assume this is being run on Roy

# fs----- imports, paths, constants -----#
import os
from os.path import join as pjoin
from collections import OrderedDict as OD
import numpy as np
import pandas as pd
from pandas import IndexSlice as idx
import matplotlib.pyplot as plt
import matplotlib as mpl

mesaruns = '/Users/troyraen/Osiris/DMS/mesaruns'
dr = mesaruns + '/RUNS_runtimeTests'

Lsun = 3.8418e33 # erg/s
Msun = 1.9892e33 # grams

figsize = (10,6)

# fe----- imports, paths, constants -----#

# fs----- mount Osiris -----#
try: # mount Osiris dir if not already
    assert os.path.isdir(mesaruns)
    print('Osiris dir is already mounted.')
except:
    mounto = int(input('Do you want to mount Osiris? (default 1 = yes)') or 1)
    if mounto == 1:
        try:
            print('Mounting Osiris.')
            os.system("sshfs tjr63@osiris-inode01.phyast.pitt.edu:/home/tjr63 /Users/troyraen/Osiris/")
            assert os.path.isdir(mesaruns)
        except:
            print('Osiris remote mount failed.')
            print('Make sure Pulse Secure is connected and Korriban is not already mounted.')
            print('If Osiris is mounted, check that this path is valid: {}.'.format(mesaruns))
            raise
# fe----- mount Osiris -----#

# fs----- load data -----#

def load_all_data(dr=dr, rk='_default_plus_DM'):
    """ rk: run_key
    """
    rcdict, pilist = {}, []
    for threads in os.listdir(pjoin(dr,rk)):
        t = int(threads.strip('threads'))
        for sim in os.listdir(pjoin(dr,rk,threads)):
            for cb in os.listdir(pjoin(dr,rk,threads,sim)):
                if cb[0] != 'c': continue
                for mdir in os.listdir(pjoin(dr,rk,threads,sim,cb)):
                    dir = pjoin(dr,rk,threads,sim,cb,mdir)
                    m = float('.'.join(mdir.strip('m').split('p')))
                    dfkey = (rk, t, int(sim[-1]))
                    print()
                    print(f'doing {dfkey}')

                    # get runtime, etc. from STD.out as Series
                    rcs = get_STDout_run_characteristics(pjoin(dir,'LOGS/STD.out'))
                    rcs['cb'], rcs['mass'] = int(cb[-1]), m

                    # Get profiles.index data
                    pidf = load_pidf(pjoin(dir,'LOGS/profiles.index'))
                    pidf['run_key'], pidf['threads'], pidf['sim'] = dfkey
                    pidf['cb'], pidf['mass'] =  int(cb[-1]), m
                    pilist.append(pidf.set_index(['run_key','threads','sim']))
                    # Set more run characteristics
                    rcs['priorities'] = pidf.priority
                    rcs['end_priority'] = pidf.loc[pidf.priority>90,'priority'].min()
                    rcs['end_model'] = pidf.model_number.max()

                    # save the series
                    rcdict[dfkey] = rcs

    pi_df = pd.concat(pilist, axis=0).sort_index()
    rcdf = pd.DataFrame.from_dict(rcdict, orient='index').sort_index()
    rcdf.index.names = ('run_key','threads','sim')
    rcdf.rename(columns={'runtime (minutes)':'runtime'}, inplace=True)
    rcdf.fillna(value=-1, inplace=True)
    rcdf = rcdf.astype({'runtime':'float32', 'retries':'int32',
                        'backups':'int32', 'steps':'int32'})

    return rcdf, pi_df


def load_pidf(pipath):
    """ Loads single profiles.index file. """
    pidx_cols = ['model_number', 'priority', 'profile_number']
    pidf = pd.read_csv(pipath, names=pidx_cols, skiprows=1, header=None, sep='\s+')
    return pidf


def get_STDout_run_characteristics(STDpath):
    with open(STDpath) as fin:
        have_runtime, have_termcode = False, False
        cols, vals = [], []
        for l, line in enumerate(reversed(fin.readlines())):
            try: # get runtime, etc.
                if line.split()[0] == 'runtime':
                    have_runtime = True
                    cl,vl = line.strip().replace('steps','steps:').split(':')
                    cols = cols + cl.split(',')
                    cols = [c.strip() for c in cols]
                    vals = vals + vl.split()
                    # rcs = pd.Series(data=vals,index=cols)
            except:
                pass
            try: # get termination code
                ln = line.strip().split(':',1)
                if ln[0] == 'termination code':
                    have_termcode = True
                    cols = cols + ['termCode']
                    vals = vals + [ln[1].strip()]
            except:
                pass
            if have_runtime and have_termcode:
                print('have runtime and termcode')
                break
            if l > 100:
                print('l > 100, leaving STD.out')
                break

    if len(cols)>0:
        rcs = pd.Series(data=vals,index=cols)
        rcs['finished'] = True if have_runtime else False
        # rcdict[dfkey] = rcs
    else:
        print('rcs failed')
        rcs = pd.Series(data=(False),index=['finished'])

    return rcs

# fe----- load data -----#


# fs----- plots -----#
def plot_avg_runtimes(rcdf, save=None):

    mean = rcdf.groupby(level='threads').mean()
    mean.reset_index().plot('threads','runtime', kind='scatter', grid=True)

    plt.ylabel('runtime [min]')
    if save is not None: plt.savefig(save)
    plt.show(block=False)

    return None




# fe----- plots -----#
