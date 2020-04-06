# paths assume this is being run on Roy

# fs----- imports, paths, constants, cmaps -----#
import os
from os.path import join as pjoin
from collections import OrderedDict as OD
import numpy as np
import pandas as pd
from pandas import IndexSlice as idx
import itertools
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import ListedColormap
from matplotlib.ticker import AutoMinorLocator
import seaborn as sns

runsdir = '/RUNS_defDM'
# Use if running on Roy
mesaruns = '/Users/troyraen/Osiris/DMS/mesaruns'
dr = mesaruns + runsdir
# Use if running on Osiris
mesarunsO = '/home/tjr63/DMS/mesaruns'
drO = mesarunsO + runsdir

Lsun = 3.8418e33 # erg/s
Msun = 1.9892e33 # grams

figsize = (10,6)

masses_scheduled = [0.85,0.9,0.95] + list(np.round(np.arange(1.0,5.05,0.05),2))
priority_dict = {   99: 'ZAMS',
                    98: 'IAMS',
                    97: f'log h1$_c$ < -1',
                    96: f'log h1$_c$ < -2',
                    95: f'log h1$_c$ < -3',
                    94: 'TAMS',
                    93: 'He ignition',
                    92: f'log he4$_c$ < -3',
                }
start_center_h1 = 0.7155
h1cuts = OD([('ZAMS',start_center_h1 - 0.0015),
             ('IAMS',0.3),
             ('H-1',1e-1),
             ('H-2',1e-2),
             ('H-3',1e-3),
             ('H-4',1e-4),
             ('H-6',1e-6),
             ('TAMS',1e-12)])
use_hcols = ['model_number', 'star_age', 'star_mass', 'log_dt',
            'log_LH', 'log_Teff','log_L',
            'center_h1', 'center_he4', 'log_center_T','log_center_Rho','wimp_temp',
            'rel_error_in_energy_conservation','error_in_energy_conservation',
            'num_iters','total_energy']


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
            \n\t1 = yes (Roy), default \
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
# instead, use bash script manually on Osiris
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

def load_profile(ppath):

    pdf = pd.read_csv(ppath, header=4, sep='\s+')

    return pdf

def load_profiles(pidf,masses,cboosts,priorities,max_mod=False):
    """ Calls load_profile() for all combinations of
        masses, cboosts, priorities
        max_mod == True will load the max model number if a priority doesn't
                    exist for a given mass+cb
    """
    pi = pidf.reset_index()
    pdf_lst = []
    pnum = 'profile_number'
    for mass in masses:
        mstr = mass_encode(mass,'str')
        for cb in cboosts:
            for EEP in priorities:
                try:
                    icols = ['cb','mass','priority']
                    i = idx[cb,mass,EEP]
                    p = pi.set_index(icols).sort_index().loc[i,pnum].iloc[0]
                    ppath = pjoin(drO,f'c{cb}',mstr,'LOGS',f'profile{p}.data')

                    pdf = load_profile(ppath)
                    pdf.rename(columns={'mass':'mass_coord'}, inplace=True)
                    pdf['cb'], pdf['mass'], pdf['EEP'] = cb, mass, EEP
                    pdf_lst.append(pdf.set_index(['cb','mass','EEP']))

                except:
                    if not max_mod: pass # else load the profile with max model number
                    icols = ['cb','mass','model_number']
                    i = idx[cb,mass,:] # index with max model number with iloc[-1]
                    p = pi.set_index(icols).sort_index().loc[i,pnum].iloc[-1]
                    ppath = pjoin(drO,f'c{cb}',mstr,'LOGS',f'profile{p}.data')

                    pdf = load_profile(ppath)
                    pdf.rename(columns={'mass':'mass_coord'}, inplace=True)
                    pdf['cb'], pdf['mass'], pdf['EEP'] = cb, mass, p
                    pdf_lst.append(pdf.set_index(['cb','mass','EEP']))

    pdf = pd.concat(pdf_lst, axis=0).sort_index()
    return pdf

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

    try:
        n = pidf.loc[pidf.priority==99,'profile_number']
        if len(n)>1: print(f'\n*** Multiple priority 99 controls files in {Lpath} ***\n')
        cpath = pjoin(Lpath,f'controls{n.iloc[0]}.data')
        cdic[tuple(list(dfkey)+[99])] = load_controls(cpath)
    except:
        print(f'profile with priority==99 could not be loaded from {Lpath}')

    try:
        n = pidf.loc[pidf.model_number==pidf.model_number.max(),'profile_number']
        if len(n)>1: print(f'\n*** Multiple max model_number controls files in {Lpath} ***\n')
        cpath = pjoin(Lpath,f'controls{n.iloc[0]}.data')
        cdic[tuple(list(dfkey)+['max_mod'])] = load_controls(cpath)
    except:
        print(f'profile with max model number could not be loaded from {Lpath}')

    cdf = pd.DataFrame.from_dict(cdic, orient='index')
    return cdf

def load_history(hpath):
    """ Loads single history.data file. """
    return pd.read_csv(hpath, header=4, sep='\s+', usecols=use_hcols)

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
    rcs['model_max'] = h.model_number.max()
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

def load_all_data(dr=dr, get_history=True, use_reduc=True, mods=None, skip=None):
    """
        mods = list of strings, ['m1p0c6']. Models NOT IN this list will be skipped.
        skip = list of strings, ['m1p0c6']. Models IN this list will be skipped.
        use_reduc == False will NOT use the __history.data__ reduced version
    """
    hlist, pilist, clist, rcdict = [], [], [], {}
    for cb in os.listdir(dr):
        if cb[0] != 'c': continue
        for mdir in os.listdir(pjoin(dr,cb)):
            rk = mdir.split('_',1)[-1]
            if rk.split('_')[0] == 'ow': continue # skip 'ow' dirs
            # skip dirs not in mods
            if mods is not None:
                if ''.join([mdir,cb]) not in mods: continue
            # skip dirs in skip
            if skip is not None:
                if ''.join([mdir,cb]) in skip: continue

            dir = pjoin(dr,cb,mdir)
            m = mass_encode(mdir,'float')
            # m = float('.'.join(mdir.split('_')[0].strip('m').split('p')))
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
                hpath = hrd if (os.path.exists(hrd) and use_reduc) else hd
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
    try:
        rcdf = rcdf.astype({'runtime':'float32', 'retries':'int32',
                            'backups':'int32', 'steps':'int32',
                            'model_max':'int32'})
    except: pass

    return hdf, pi_df, c_df, rcdf

def load_dt_root_errors(dr=dr, mods=[]):
    """ mods = list of strings, e.g. 'm1p0c6'. Models not in this list will be skipped.

    Returns:
        reddtdf:    codes and model numbers where dt was reduced
        probTxdf:   models where STD.out says 'Tx set to center_T'
                    if this is not an empty dataframe, this is a PROBLEM
    """
    reddt_list, probTx_list = [], []
    for cb in os.listdir(dr):
        if cb[0] != 'c': continue
        for mdir in os.listdir(pjoin(dr,cb)):
            rk = mdir.split('_',1)[-1]
            if rk.split('_')[0] == 'ow': continue # skip 'ow' dirs
            if ''.join([mdir,cb]) not in mods: continue # skip dirs not in mods

            m = mass_encode(mdir,'float')
            # m = float('.'.join(mdir.strip('m').split('p')))
            dir = pjoin(dr,cb,mdir)

            reddtdf, probTxdf = get_STDout_dt_root_warnings(pjoin(dir,'LOGS/STD.out'))
            reddtdf['cb'], probTxdf['cb'] = int(cb[1]), int(cb[1])
            reddtdf['mass'], probTxdf['mass'] = m, m

            reddt_list.append(reddtdf), probTx_list.append(probTxdf)

    reddtdf, probTxdf = pd.concat(reddt_list), pd.concat(probTx_list)
    reddtdf.set_index(['cb','mass'], inplace=True)
    probTxdf.set_index(['cb','mass'], inplace=True)

    return reddtdf, probTxdf

def get_STDout_dt_root_warnings(STDpath):
    dtcodes, dtmods = [], []
    TxeqTc = []
    with open(STDpath) as fin:
        for l, line in enumerate(fin.readlines()):
            if line[:21] == 'reduce dt because of ':
                ln = line[21:].split()
                # set code to all words, set mod to first int
                code = ln.pop(0)
                while not ln[0].isnumeric():
                    code = ' '.join([code,ln.pop(0)])
                dtcodes.append(str(code)), dtmods.append(float(ln[0]))
            elif line[:28] == '**** Tx set to center_T ****':
                TxeqTc.append(line.split('problem model ')[1])

    reddtdf = pd.DataFrame({'code':dtcodes, 'model':dtmods})
    # reddtdf.astype({'model':})
    if len(TxeqTc)>0:
        probTxdf = pd.DataFrame({'model':TxeqTc})
    else:
        probTxdf = pd.DataFrame({'model':(np.nan)}, index=['idx'])

    return reddtdf, probTxdf

def fix_negative_runtimes(rcdf, rtfix=None):
    """ Fix runtimes with negative number in STD.out
        Actual time calculated from file timestamps

        rtfix = {index: runtime in days (gets converted to minutes)}
    """

    minday = 24*60

    if rtfix is None:
        rtfix = {
                    idx[1,1.15]: 17,
                    idx[1,1.2]: 18.3,
                    idx[2,1.1]: 22.5,
                    idx[4,1.1]: 14.6,
                    idx[4,1.25]: 20.9,
                    idx[6,1.0]: 19,
                }

    for i, val in rtfix.items():
        rcdf.loc[i,'runtime'] = val*minday
        print(f'fixed {i}')

    return rcdf


# fe----- load data -----#

# fs------ plots and functions ------#
def mass_encode(mass,get):
    """ get == 'str':   converts mass (float) to a string (e.g. m1p00)
        get == 'float': converts mass (string) to a float (e.g. 1.0)
    """
    if get == 'str':
        mstr = f'm{int(mass)}p{int((mass%1)*10)}'
        if len(mstr) == 4: mstr = f'{mstr}0'
        return mstr

    elif get == 'float':
        m = float('.'.join(mass.strip('m').split('p')))
        return m

def MSmodels(hdf):
    """ hdf should be of a single mass and cb
        can use in hdf.groupby().apply()
    """
    ch1 = hdf.center_h1
    mods = hdf.loc[((ch1<h1cuts['ZAMS'])&(ch1>h1cuts['TAMS'])),'model_number']
    enter, leave = mods.min(), mods.max()
    final = hdf.model_number.max()
    return pd.Series({'enter':enter, 'leave':leave, 'final':final})

def get_mods_not_done(cmidx):
    """ cmidx = a multi-index (cb,  mass) from any of the dfs
        returns models that are queued, waiting to be run
    """
    schd = set(itertools.product([i for i in range(7)],masses_scheduled))
    not_done = schd - set(cmidx)
    return not_done

def plot_runtimes(rcdf, save=None):

    rc = rcdf.loc[rcdf.runtime>0,:].copy()
    plt.figure(figsize=figsize)
    ax = plt.gca()
    kwargs = {  'ax':ax,
                'marker':'o'
                }
    for cb, df in rc.reset_index('mass').groupby(level='cb'):
        d = df.sort_values('mass')
        clr = cbcmap(cb)
        d.plot('mass','runtime',label=cb,c=clr, **kwargs)

    # add dots repping models not yet run
    not_done = get_mods_not_done(rcdf.index)
    dfr = pd.DataFrame({'runtime':1e6}, index=not_done)
    dfr.index.rename(['cb','mass'], inplace=True)
    dfr['runtime'] = dfr.runtime * (dfr.reset_index().cb.values + 1)
    for cb, df in dfr.reset_index('mass').groupby(level='cb'):
        d = df.sort_values('mass')
        clr = cbcmap(cb)
        plt.plot(d.mass,d.runtime, 'o-', c=clr)

    # pvt = {'index':'mass','columns':'cb','values':'runtime'}
    # clr = [cbcmap(i) for i in range(7)]
    # df.reset_index().pivot(**pvt).plot(**kwargs,color=clr)

    plt.xlabel('Mass')
    plt.ylabel('Runtime [min]')
    plt.loglog()
    plt.grid(True)
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

def plot_HR(hdf, color=None, title=None, save=None):
    """ if color == 'dt', will make scatter plot and color using dt
    """

    kwargs = {}

    if color == 'dt':
        kwargs['cmap'] = plt.get_cmap('afmhot')
        kwargs['vmin'], kwargs['vmax'] = -12, 15
        kwargs['s'] = 3

    gb = hdf.groupby(level=['cb','mass'])
    nrows, ncols = int(np.ceil(len(gb)/3)), 3
    fig, axs = plt.subplots(nrows=nrows,ncols=ncols,
                            # sharex=True,sharey=True,
                            figsize=(8,3*nrows))
    if nrows!=1: axs = axs.flatten()


    for a, ((cb,mass), df) in enumerate(gb):
        ax = axs[a]
        df = df.loc[df.star_age>1e6,:]
        endHe4 = df.loc[df.star_age==df.star_age.max(),'center_he4']
        lbl = f'm{mass} c{cb}\ncenterHe4_end = {endHe4.iloc[0]:.2f}'

        if color == 'dt':
            c = df.log_dt
            scat = ax.scatter(df.log_Teff, df.log_L, label=lbl, c=c, **kwargs)
        else:
            kwargs['ax'] = ax
            # print(kwargs)
            df.plot('log_Teff','log_L', label=lbl, **kwargs)

        # indicate where age > 10 and 13 Gyr
        df = df.reset_index().set_index(['cb','mass','model_number'])
        sa = df.star_age
        for t,m in zip([10e9, 13e9],['s','X']):
            ags = {'marker':m, 's':100, 'c':'k', 'label':f'{t/1e9:.0f}Gyr'}
            try:
                i = df[sa.gt(t)].sort_values('star_age').index[0]
                ax.scatter(df.loc[i,'log_Teff'],df.loc[i,'log_L'], **ags)
            except:
                pass

        ax.invert_xaxis()
        ax.set_xlabel(r'log T$_{eff}$ / K')
        ax.legend(loc=3)
    axs[0].set_ylabel(r'log L / L$_\odot$')
    if color == 'dt':
        cbar = plt.colorbar(scat)#, cax=cbax, orientation='horizontal')
        cbar.set_label('log dt')
    if title is not None: plt.suptitle(title)

    plt.tight_layout()
    if save is not None: plt.savefig(save)
    plt.show(block=False)
    return None

def plot_reddt(reddtdf, title=None, save=None):

    codemap = dict(list(enumerate(reddtdf.code.unique()))) # int: code
    inv_codemap = {v: k for k, v in codemap.items()} # code: int
    df = reddtdf.copy()
    df['code_idx'] = df.code.map(inv_codemap)

    gb = df.groupby(level=['cb','mass'])
    fig, axs = plt.subplots(nrows=len(gb), ncols=1, sharex=True)
    # ax = plt.gca()
    # kwargs = {  'ax': ax,
    #             'logx': True,
    #             }
    for a, ((cb,mass), d) in enumerate(gb):
        lbl = f'm{mass} c{cb}'
        axs[a].scatter(d.model, d.code_idx, label=lbl)

        axs[a].legend()

    if save is not None: plt.savefig(save)
    plt.show(block=False)

    return None

def plot_profiles(pdf,ax):
    """ pdf is single mass, cb, profile number
    """
    p = pdf.copy()
    p['nuc_H'] = p.pp + p.cno
    p.rename(columns={'extra_heat':'xheat'}, inplace=True)

    cols = []
    for c in [  'xheat', 'nuc_H',
                # 'eps_nuc',
                ]:
        mx = p[c].abs().max()
        nam = '/'.join([c,f'{mx:.1}'])
        cols.append(nam)
        p[nam] = p[c]/mx

    cols.append('x')
    args = {'ax':ax, 'alpha':1}
    p.plot(x='mass_coord', y=cols, **args)
    p.plot(x='mass_coord', y=cols[0], kind='scatter', ax=ax)
    ax.set_ylabel('')

    return None

def plot_profiles_all(pdf):

    gb = pdf.groupby(level=['cb','mass','EEP'])
    l = len(gb)
    nr, nc = int(np.ceil(l/3)), 3
    fig, axs = plt.subplots(nrows=nr,ncols=nc, figsize=(10,3*nr),sharex=True)
    axs = axs.flatten()
    for a, (k, p) in enumerate(gb):
        ax = axs[a]
        plot_profiles(p,ax)
        eep = priority_dict[k[2]] if k[2] in priority_dict.keys() else f'maxmodel(p{k[2]})'
        ax.set_title(f'm{k[1]}c{k[0]} {eep}')
    ax.set_xlim(-0.01,0.6)

    return None


# fe------ plots and functions ------#
