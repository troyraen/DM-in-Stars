# paths assume this is being run on Roy

# fs----- imports, paths, constants, cmaps -----#
import os
from os.path import join as pjoin
from collections import OrderedDict as OD
import numpy as np
import pandas as pd
from pandas import IndexSlice as idx
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import ListedColormap

mesaruns = '/Users/troyraen/Osiris/DMS/mesaruns'
dr = mesaruns + '/RUNS_defDM'
run_keys = []

Lsun = 3.8418e33 # erg/s
Msun = 1.9892e33 # grams

figsize = (10,6)

priority_dict = {   99: 'ZAMS',
                    98: 'IAMS',
                    97: f'log h1$_c$ < -1',
                    96: f'log h1$_c$ < -2',
                    95: f'log h1$_c$ < -3',
                    94: 'TAMS',
                    93: 'He ignition',
                    92: f'log he4$_c$ < -3',
                }

def normalize_RGB(rgb_tuple):
    """ rgb_tuple assumed to be of length 3, values in range [0,255]
        Returns tuple with rgb_tuple values divided by 255,
            with 1 appended to end of tuple (needed for alpha in colormaps)
    """

    rbgnorm = 255.0
    rgb_scaled = [ c/rbgnorm for c in rgb_tuple ]
    rgb_scaled.append(1.0)
    return tuple(rgb_scaled)

c0c = normalize_RGB((169,169,169))
c1c = normalize_RGB((127,205,187))
c2c = normalize_RGB((65,182,196))
c3c = normalize_RGB((29,145,192))
c4c = normalize_RGB((34,94,168))
c5c = normalize_RGB((37,52,148))
c6c = normalize_RGB((8,29,88))
carr = (c0c,c1c,c2c,c3c,c4c,c5c,c6c)
cbcmap = ListedColormap(carr)

# fe----- imports, paths, constants, cmaps -----#

# fs----- mount Osiris -----#
try: # mount Osiris dir if not already
    assert os.path.isdir(mesaruns)
    print('Osiris dir is already mounted.')
except:
    msg = 'Do you want to mount Osiris? \
            \n\t1 = yes (Roy) \
            \n\telse no\n'
    mounto = int(input(msg) or 1)
    if mounto in [1]:
        localpath = '/Users/troyraen/Osiris/' # if mounto == 1 else '/home/tjr63/Osiris/'
        try:
            print('Mounting Osiris.')
            os.system(f"sshfs tjr63@osiris-inode01.phyast.pitt.edu:/home/tjr63 {localpath}")
            assert os.path.isdir(mesaruns)
        except:
            print('Osiris remote mount failed.')
            print('Make sure Pulse Secure is connected and Korriban is not already mounted.')
            print('If Osiris is mounted, check that this path is valid: {}.'.format(mesaruns))
            raise
# fe----- mount Osiris -----#

# fs----- load data -----#
def reduc_all_LOGS(RUNSdir=None):
    """ runs check_for_reduc() on all dirs in mesaruns/dr (defined above)
    """
    if RUNSdir is None: RUNSdir = pjoin(mesaruns,dr)
    for cbdir in os.listdir(RUNSdir):
        for mdir in os.listdir(pjoin(RUNSdir,cbdir)):
            dir = pjoin(RUNSdir,cbdir,mdir,'LOGS')
            check_for_reduc(dir)

    return None

# check_for_reduc() is too slow to use on Roy
def check_for_reduc(LOGSpath, max_fsize=500.0):
    """ Checks the following files in LOGSpath dir and creates a reduced
        version if file size > max_fsize [MB] (and does not already exist):
            STD.out
            history.data
    """

    smax = max_fsize*1024*1024 # [bytes]
    z = zip(['STD.out', 'history.data'],['STD_reduc.out', 'history_reduc.data'])
    for fil, rfil in z:
        typ = fil.split('.')[0]

        OGp, rp = pjoin(LOGSpath,fil), pjoin(LOGSpath,rfil)
        if os.path.exists(rp): continue

        if os.path.getsize(OGp) > smax:
            print(f'Reducing {typ} at {OGp}')
            if typ == 'STD':
                os.system(f"tail -n 100 '{OGp}' > '{rp}'")
            elif typ == 'history':
                os.system(f'../bash_scripts/data_reduc.sh {LOGSpath}')

    return None

def load_pidf(pipath):
    """ Loads single profiles.index file. """
    pidx_cols = ['model_number', 'priority', 'profile_number']
    pidf = pd.read_csv(pipath, names=pidx_cols, skiprows=1, header=None, sep='\s+')
    return pidf

def load_controls(cpath):
    """ Loads a single controls#.data file to a Series """

    lines_with_known_probs = [
                                'DIFFUSION_CLASS_REPRESENTATIVE',
                                'DIFFUSION_CLASS_A_MAX',
                                'DIFFUSION_CLASS_TYPICAL_CHARGE',
                                ]

    cs = pd.Series()
    with open(cpath) as fin:
        for l, line in enumerate(fin.readlines()):
            # print(line, len(line))
            if line.strip() == '&CONTROLS': continue # skip first line
            try:
                name, val = line.split('=',1)
            except:
                if line.strip() == '/':
                    continue
                else: # add to val of previous line
                    newval = line.split(',')
                    val = val + newval
                    if name not in lines_with_known_probs:
                        print(f'added new line values: {newval} \nto previous line #{l}: {name} \n')
            else:
                name = name.strip()
                val = val.split(',')

            if val[-1] == '\n': del val[-1]
            if len(val) == 1: val = val[0].strip()

            cs[name] = val

    return cs

def load_be_controls(Lpath, pidf, dfkey):
    """ Loads 2 controls#.data files (beginning and ending controls):
            1. priofile priority == 99 (enter MS)
            2. profiles model number == max

        Lpath = path to LOGS dir
        pidf = DF of profiles.index, as returned by load_pidf()
        dfkey = (multi-)index for this run (for the DataFrame)
    """

    cdic = {}

    n = pidf.loc[pidf.priority==99,'profile_number']
    if len(n)>1: print(f'\n*** Multiple priority 99 controls files in {Lpath} ***\n')
    cpath = pjoin(Lpath,f'controls{n.iloc[0]}.data')
    cdic[tuple(list(dfkey)+[99])] = load_controls(cpath)

    n = pidf.loc[pidf.model_number==pidf.model_number.max(),'profile_number']
    if len(n)>1: print(f'\n*** Multiple max model_number controls files in {Lpath} ***\n')
    cpath = pjoin(Lpath,f'controls{n.iloc[0]}.data')
    cdic[tuple(list(dfkey)+['max_mod'])] = load_controls(cpath)

    cdf = pd.DataFrame.from_dict(cdic, orient='index')
    return cdf

def load_history(hpath):
    """ Loads single history.data file. """
    return pd.read_csv(hpath, header=4, sep='\s+')

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

def get_history_run_characteristics(rcs, h):
    """ rcs = Series, as returned by get_STDout_run_characteristics()
        h = history DataFrame (single run)
    """
    rcs['num_iters_tot'] = h.num_iters.sum()
    rcs['log_max_rel_energy_error'] = \
            np.log10(h.rel_error_in_energy_conservation.abs().max())
    rcs['log_cum_rel_energy_error'] = \
            np.log10(np.abs(h.error_in_energy_conservation.sum() \
                      /h.sort_values('star_age').total_energy.iloc[-1]))
    rcs['log_dt_avg'] = np.log10((10**h.log_dt).mean())
    rcs['log_dt_min'] = h.log_dt.min()
    rcs['log_dt_max'] = h.log_dt.max()
    rcs['hottest_logTeff'] = h.log_Teff.max()
    rcs['center_h1_end'] = h.sort_values('star_age').center_h1.iloc[-1]
    rcs['center_he4_end'] = h.sort_values('star_age').center_he4.iloc[-1]
    try:
        rcs['MStau'], __, mod_leave = get_MStau(h)
        rcs['logL_leaveMS'] = h.loc[h.model_number==mod_leave,'log_L'].iloc[0]
    except:
        print('MStau failed')
        pass

    return rcs

def get_MStau(hdf):
    """ hdf is DataFrame of single star
    """
    d = hdf.reset_index().sort_values('star_age')

    mod_enter = d.loc[d.center_h1<(d.center_h1.iloc[0]-0.0015),'model_number'].iloc[0]
    enter = d.loc[d.model_number==mod_enter,'star_age'].iloc[0]
    # print(f'mod enter {mod_enter}, enter {enter}')

    mod_leave = d.loc[d.center_h1<0.001,'model_number'].iloc[0]
    leave = d.loc[d.model_number==mod_leave,'star_age'].iloc[0]
    # print(f'mod leave {mod_leave}, leave {leave}')

    MStau = leave-enter
    # print(f'MStau {MStau}')

    return MStau, mod_enter, mod_leave

def load_all_data(dr=dr, get_history=True):
    """
    """
    hlist, pilist, clist, rcdict = [], [], [], {}
    for cb in os.listdir(dr):
        if cb[0] != 'c': continue
        for mdir in os.listdir(pjoin(dr,cb)):
            rk = mdir.split('_',1)[-1]
            if rk.split('_')[0] == 'ow': continue # skip 'ow' dirs

            dir = pjoin(dr,cb,mdir)
            m = float('.'.join(mdir.split('_')[0].strip('m').split('p')))
            dfkey = (int(cb[-1]), m)
            print()
            print(f'doing {dfkey}')

            # get runtime, etc. from STD.out as Series
            sd = pjoin(dir,'LOGS/STD.out')
            srd = pjoin(dir,'LOGS/STD_reduc.out') # use if exists
            spath = srd if os.path.exists(srd) else sd
            rcs = get_STDout_run_characteristics(spath)

            # Get profiles.index data
            pidf = load_pidf(pjoin(dir,'LOGS/profiles.index'))
            pidf['cb'], pidf['mass'] = dfkey
            pilist.append(pidf.set_index(['cb','mass']))
            # Set more run characteristics
            # rcs['priorities'] = pidf.priority
            rcs['end_priority'] = pidf.loc[pidf.priority>90,'priority'].min()

            # Get controls#.data as DataFrame
            cdf = load_be_controls(pjoin(dir,'LOGS'), pidf, dfkey)
            clist.append(cdf)

            # Get history.data
            if get_history:
                hd = pjoin(dir,'LOGS/history.data')
                hrd = pjoin(dir,'LOGS/history_reduc.data') # use if exists
                hpath = hrd if os.path.exists(hrd) else hd
                h = load_history(hpath)
                h.set_index('model_number', inplace=True)
                h['profile_number'] = pidf.set_index('model_number').profile_number
                # save the history dataframe
                h['cb'], h['mass'] = dfkey
                h = h.reset_index().set_index(['cb','mass'])
                hlist.append(h)
                # Set more run characteristics
                rcs = get_history_run_characteristics(rcs, h)

            # save the series
            rcdict[dfkey] = rcs

    hdf = pd.concat(hlist, axis=0).sort_index() if get_history else []
    pi_df = pd.concat(pilist, axis=0).sort_index()
    c_df = pd.concat(clist, axis=0).sort_index()
    rcdf = pd.DataFrame.from_dict(rcdict, orient='index').sort_index()
    rcdf.index.names = ('cb','mass')
    rcdf.rename(columns={'runtime (minutes)':'runtime'}, inplace=True)
    rcdf.fillna(value=-1, inplace=True)
    rcdf = rcdf.astype({'runtime':'float32', 'retries':'int32',
                        'backups':'int32', 'steps':'int32'})

    return hdf, pi_df, c_df, rcdf

# fe----- load data -----#

# fs------ plots ------#
def plot_runtimes(rcdf, save=None):
    # all runtimes
    rc = rcdf.loc[rcdf.runtime>0,:].copy()
    rc.loc[idx[6,1.0],'runtime'] = 19*24*60 # this is a negative number in STD.out
                                            # actual time calculated from file timestamps
    plt.figure(figsize=figsize)
    ax = plt.gca()
    kwargs = {  'loglog':True,
                'grid':True,
                'ax':ax,
                # 'kind':'scatter'
                }
    for cb, df in rc.reset_index('mass').groupby(level='cb'):
        d = df.sort_values('mass')
        clr = cbcmap(cb)
        d.plot('mass','runtime',label=cb,c=clr, **kwargs)

    plt.xlabel('Mass')
    plt.ylabel('Runtime [min]')
    plt.tight_layout()
    if save is not None: plt.savefig(save)
    plt.show(block=False)

    return None

def plot_log_dt(hdf, save=None):

    plt.figure()
    ax = plt.gca()
    kwargs = {  'ax': ax,
                'logx': True,
                }
    for (cb,mass), df in hdf.groupby(level=['cb','mass']):
        lbl = f'm{mass} c{cb}'
        df.loc[df.star_age>1e6,:].plot('star_age','log_dt', label=lbl, **kwargs)

    plt.legend()

    if save is not None: plt.savefig(save)
    plt.show(block=False)
    return None

def plot_HR(hdf, title=None, save=None):

    plt.figure()
    ax = plt.gca()
    kwargs = {  'ax': ax
                }

    for (cb,mass), df in hdf.groupby(level=['cb','mass']):
        df = df.loc[df.star_age>1e6,:]
        lbl = f'm{mass} c{cb}'
        df.plot('log_Teff','log_L', label=lbl, **kwargs)

    ax.invert_xaxis()
    plt.xlabel(r'log T$_{eff}$ / K')
    plt.ylabel(r'log L / L$_\odot$')
    plt.legend()
    if title is not None: plt.title(title)

    if save is not None: plt.savefig(save)
    plt.show(block=False)
    return None


# fe------ plots ------#
