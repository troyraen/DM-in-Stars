# paths assume this is being run on Roy

# fs----- imports, paths, constants -----#
from warnings import warn as _warn
import os
# import subprocess as subp
from os.path import join as pjoin
from collections import OrderedDict as OD
import numpy as np
import pandas as pd
from pandas import IndexSlice as idx
import matplotlib.pyplot as plt
import matplotlib as mpl

mesaruns = '/Users/troyraen/Osiris/DMS/mesaruns'
dr = mesaruns + '/RUNS_runSettings'
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
def load_profiles(dirs):
    """ Returns df of all profile#.data files in dirs
                df of all profiles.index files in dirs

        Args    dirs (list): LOGS directory paths
    """

    pilst, plst = [], []

    for d in dirs:
        pidf = load_pidf(pjoin(d,'profiles.index'))
        pikey = {'mass_rk':d.split('/')[-2], 'cb':d.split('/')[-3]}
        for k,val in pikey.items():
            pidf[k] = val
        pidf.set_index([k for k in pikey.keys()]+['priority'], inplace=True)
        pilst.append(pidf)

        if not pidf.reset_index().priority.is_unique:
            _warn(f'\npriorities not unique in {pikey.values()}.'
                    '\npdf will have duplicate keys.\n')

        for pn, prior in zip(pidf.profile_number, pidf.reset_index().priority):
            pdf = pd.read_csv(pjoin(d,f'profile{pn}.data'), header=4, sep='\s+')
            pkey = pikey.copy()
            pkey['priority'] = prior
            for k,val in pkey.items():
                pdf[k] = val
            pdf.set_index([k for k in pkey.keys()]+['zone'], inplace=True)
            plst.append(pdf)

    pi_df = pd.concat(pilst, axis=0)
    p_df = pd.concat(plst, axis=0)

    return p_df, pi_df

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

def check_for_reduc(LOGSpath, max_fsize=5.0):
    """ Checks the following files in LOGSpath dir and creates a reduced
        version if file size > max_fsize [MB] (and does not already exist):
            STD.out
            history.data
    """

    smax = max_fsize/(1024*1024) # [bytes]
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

def load_all_data(dr=dr, run_key=['all'], get_history=True):
    """ run_key 'all' or list of strings (without _ prefix)
    """
    hlist, pilist, clist, rcdict = [], [], [], {}
    for cb in os.listdir(dr):
        if cb[0] != 'c': continue
        for mdir in os.listdir(pjoin(dr,cb)):
            rk = mdir.split('_',1)[-1]

            # use only dirs in run_key
            if (rk not in run_key) and (run_key!=['all']): continue
            if rk.split('_')[0] == 'ow': continue # skip 'ow' dirs

            dir = pjoin(dr,cb,mdir)
            m = float('.'.join(mdir.split('_')[0].strip('m').split('p')))
            dfkey = (rk, int(cb[-1]), m)
            print()
            print(f'doing {dfkey}')

            # reduce STD and history file sizes if needed
            check_for_reduc(pjoin(dir,'LOGS'), max_fsize=5.0)

            # get runtime, etc. from STD.out as Series
            sd = pjoin(dir,'LOGS/STD.out')
            srd = pjoin(dir,'LOGS/STD_reduc.out') # use if exists
            spath = srd if os.path.exists(srd) else sd
            rcs = get_STDout_run_characteristics(spath)

            # Get profiles.index data
            pidf = load_pidf(pjoin(dir,'LOGS/profiles.index'))
            pidf['run_key'], pidf['cb'], pidf['mass'] = dfkey
            pilist.append(pidf.set_index(['run_key','cb','mass']))
            # Set more run characteristics
            rcs['priorities'] = pidf.priority
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
                h['run_key'], h['cb'], h['mass'] = dfkey
                h = h.reset_index().set_index(['run_key','cb','mass'])
                hlist.append(h)
                # Set more run characteristics
                rcs = get_history_run_characteristics(rcs, h)

            # save the series
            rcdict[dfkey] = rcs
            # del rcs # so that the try statement above works properly

    hdf = pd.concat(hlist, axis=0).sort_index() if get_history else []
    pi_df = pd.concat(pilist, axis=0).sort_index()
    c_df = pd.concat(clist, axis=0).sort_index()
    rcdf = pd.DataFrame.from_dict(rcdict, orient='index').sort_index()
    rcdf.index.names = ('run_key','cb','mass')
    rcdf.rename(columns={'runtime (minutes)':'runtime'}, inplace=True)
    rcdf.fillna(value=-1, inplace=True)
    rcdf = rcdf.astype({'runtime':'float32', 'retries':'int32',
                        'backups':'int32', 'steps':'int32'})

    return hdf, pi_df, c_df, rcdf

# fs
# def get_paths(dr=dr, run_key=''):
#     c0path = dr+ '/c0/m1p0' + run_key + '/LOGS'
#     h0path = c0path+ '/history.data'
#     pi0path = c0path+ '/profiles.index'
#
#     c6path = dr+ '/c6/m1p0' + run_key + '/LOGS'
#     h6path = c6path+ '/history.data'
#     pi6path = c6path+ '/profiles.index'
#
#     return (c0path, h0path, pi0path, c6path, h6path, pi6path)
#
# def load_main_data(dr=dr, run_key=''):
#
#     __, h0path, pi0path, __, hpath, pipath = get_paths(dr=dr, run_key=run_key)
#
#     hdf = pd.read_csv(hpath, header=4, sep='\s+').set_index('model_number', drop=False)
#     pidx_cols = ['model_number', 'priority', 'profile_number']
#     pidf = pd.read_csv(pipath, names=pidx_cols, skiprows=1, header=None, sep='\s+')
#     # add profile numbers to hdf
#     hdf['profile_number'] = pidf.set_index('model_number').profile_number
#
#     h0df = pd.read_csv(h0path, header=4, sep='\s+').set_index('model_number', drop=False)
#     pi0df = pd.read_csv(pi0path, names=pidx_cols, skiprows=1, header=None, sep='\s+')
#     h0df['profile_number'] = pi0df.set_index('model_number').profile_number
#
#     return (hdf, pidf, h0df, pi0df)
#
# def load_profiles_df(pnums4df, cb=6, dr=dr, run_key=''):
#
#     c0path, __, __, c6path, __, __ = get_paths(dr=dr, run_key=run_key)
#
#     dfs = []
#     for p in pnums4df:
#         if cb==6: path = c6path
#         elif cb==0: path = c0path
#         ppath = path+ f'/profile{p}.data'
#         pdf = pd.read_csv(ppath, header=4, sep='\s+')
#         pdf['profile_number'] = p
#         dfs.append(pdf.set_index(['profile_number','zone']))
#     pdf = pd.concat(dfs, axis=0)
#
#     return pdf
#
# fe

def lums_dict(hdf, lums, age_cut=1e7):
    h = hdf.loc[hdf.star_age>age_cut,:]
    age = h.star_age
    try:
        L = h.luminosity
    except:
        L = 10**h.log_L
    LH = 10**h.log_LH
    LHe = 10**h.log_LHe
    Lnuc = 10**h.log_Lnuc # excludes neutrinos [Lsun]
    Lneu = 10**h.log_Lneu # power emitted in neutrinos, nuclear and thermal [Lsun]
    # try:
    Lgrav = h.eps_grav_integral # [Lsun]
    Ltneu = 10**h.log_Lneu_nonnuc # power emitted in neutrinos, thermal sources only [Lsun]
    # except:
    #     Lgrav = 0
    #     Ltneu = 0
    extra_L = h.extra_L # [Lsun]
    # LTgrav = h.total_eps_grav/Lsun # DON'T KNOW THE UNITS. probably erg/g/s

    dic = OD([
            ('age', age),
            ('L', (L, ':')),
            ('LH', (LH, '-.')),
            ('LHe', (LHe, ':')),
            ('extra_L', (extra_L, '-')),
            ('Lneu', (Lneu, ':')),
            ('Lnuc', (Lnuc, '-')),
            ('Lgrav', (Lgrav, '-')),
            ('Ltneu', (Ltneu, '-')),
            # ('LTgrav', (LTgrav, '-')),
          ])

    d = OD([])
    for key,val in dic.items():
        if key in lums:
            d[key] = dic[key]

    return d

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

def calc_delta_MStau(rcdf, c0_ref_key=None):
    """ rcdf as returned by load_all_data()
        c0_ref_key: delta_MStau for c0 models calculated w.r.t. this run_key

        Returns rcdf with delta_MStau column added
    """

    rcdf['delta_MStau'] = np.nan
    for rk, df in rcdf.groupby(level=['run_key','mass']):
        print(rk)
        for cb, d in df.groupby(level='cb'):
            # print(d)
            # break
            # print((rk[0],0,rk[1]))
            if cb==0 and c0_ref_key is not None:
                print((c0_ref_key,0,rk[1]))
                try:
                    MStau0 = rcdf.loc[(c0_ref_key,0,rk[1]),'MStau']
                except:
                    MStau0 = df.loc[(rk[0],0,rk[1]),'MStau']
            else:
                MStau0 = df.loc[(rk[0],0,rk[1]),'MStau']
            # print(MStau0)
            # print(d.MStau)
            # print((d.MStau - MStau0)/MStau0)
            # print(rk[0])
            cbkey = (rk[0],cb,rk[1])
            rcdf.loc[cbkey,'delta_MStau'] = (d.loc[cbkey,'MStau'] - MStau0)/MStau0

    return rcdf


def control_diff(cdf):
    c = cdf.copy()
    for col in c.columns:
        # transform lists to strings
        c[col] = c[col].apply(lambda x: ','.join(x) if type(x)==list else x)
        if len(c[col].unique()) == 1:
            c.drop(col,inplace=True,axis=1)
    return c

# fe----- load data -----#

# fs----- plots -----#
def plot_pidf(pidf, save=None):
    plt.figure()

    for k,df in pidf.groupby(level=['run_key','cb','mass']):
        dfm = df.model_number
        p = df.loc[((df.priority>90)|(dfm==dfm.max())),:].sort_values('model_number')
        p.loc[p.priority<90,'priority'] = 90
        plt.plot(p.model_number,p.priority, label=k)
        plt.scatter(p.model_number,p.priority, s=50)
        # p = p.loc[df.priority==]

    plt.legend()
    plt.xlabel('model number')
    plt.ylabel('priority')
    # plt.tight_layout()
    if save is not None: plt.savefig(save)
    plt.show(block=False)

    return None

def plot_rcdf(rcdf, cols=None, save=None):

    rcg = rcdf.reset_index(level='mass').groupby(level=['run_key','cb'])

    if cols is None:
        cols = ['runtime', 'retries', 'backups', 'steps', 'log_dt_min',
                'end_priority', 'center_h1_end', 'center_he4_end',
                'log_max_rel_energy_error', 'log_cum_rel_energy_error']

    cmap = mpl.cm.get_cmap('tab20')

    f, axs = plt.subplots(sharex=True, nrows=len(cols),ncols=1, figsize=figsize)
    for i, c in enumerate(cols):
        for j, (k, df) in enumerate(rcg):
            clr = cmap(j/len(rcg))

            d = df.loc[df.finished==True,:]
            y = d[c] if c!='termCode' else map_termCodes(d.termCode)
            axs[i].scatter(d['mass'],y, label=k, c=clr)

            d = df.loc[df.finished==False,:]
            y = d[c] if c!='termCode' else map_termCodes(d.termCode)
            axs[i].scatter(list(d['mass']),list(y), c=clr, edgecolors='k')

            ls = ':' if k[1]==0 else '-'
            y = df[c] if c!='termCode' else map_termCodes(df.termCode)
            axs[i].plot(list(df['mass']),list(y), c=clr, ls=ls)

        axs[i].set_ylabel(c)

    axs[0].legend()
    axs[-1].set_xlabel('mass')
    if save is not None: plt.savefig(save)
    plt.show(block=False)

def map_termCodes(termCodes):
    dic = {
            ' HB_limit':2,
            'HB_limit':2,
            ' stop_at_TP':1,
            'stop_at_TP':1,
            -1:-1,
            ' min_timestep_limit':-2,
            'min_timestep_limit':-2,
        }

    return termCodes.map(dic)

def plot_rcdf_finished(rcdf, save=None):

    df = rcdf.reset_index()
    kwg = {#'s': df.cb*10+100,# 'alpha':0.25,
           'edgecolors':'0.5', 'facecolors':'none'}

    plt.figure()

    d = df.loc[df.finished==True,:]
    # kwg['c'] = d.cb
    kwg['s'] = d.cb*10+10
    k = d.run_key.apply(convert_runkey)
    plt.scatter(d.mass,k, **kwg)

    d = df.loc[df.finished==False,:]
    kwg['edgecolors'], kwg['linewidths'] = 'k', 3
    kwg['c'], kwg['s'] = d.cb, d.cb*50+10
    k = d.run_key.apply(convert_runkey)
    plt.scatter(d.mass,k, **kwg)
    try:
        l = f'{d.log_dt_min:0.2f}'
        plt.annotate((d.mass,k), l)
    except:
        pass

    plt.xlabel('mass')
    plt.ylabel('run_key')

    # plt.legend()
    plt.tight_layout()
    if save is not None: plt.savefig(save)
    plt.show(block=False)

def convert_runkey(run_key):
    """ run_key is a string
    """
    rk = run_key.split('mist',1)
    k = -1 if (len(rk) == 1) else int(rk[-1].split('m')[0])

    return k

def plot_delta_MStau(rcdf, title='', save=None):

    plt.figure()

    for rk, df in rcdf.reset_index('mass').groupby(level=['run_key','cb']):
        ls = ':' if rk[1]==0 else '-'
        plt.plot(df.mass,df.delta_MStau,label=rk, ls=ls)
        plt.scatter(list(df.mass),list(df.delta_MStau))

    plt.xlabel('Stellar Mass')
    plt.ylabel('delta MStau')
    plt.title(title)
    plt.legend()
    plt.semilogx()
    plt.grid()
    plt.show(block=False)
    if save is not None: plt.savefig(save)

    return None


def plot_profile(pdf, col, qlim=1.0):

    nrows = 99-91 # number of priorities we care about
    ncols = 1
    f, axs = plt.subplots(nrows=nrows,ncols=ncols, sharex=True, sharey=True,
                            figsize=(10,8))
    # plt.subplots_adjust(hspace=0.75)

    p = pdf.loc[pdf.q<qlim,:].sort_index().loc[idx[:,:,92:99,:],:]
    for a, (prior, df) in enumerate(p.groupby(level='priority')):
        ax = axs[a]
        ax.set_title(priority_dict[prior])
        ax.set_ylabel(col)

        for cb, d in df.groupby(level='cb'):
            if col == 'np' and cb=='c0': continue

            kwgs = {'ax':ax, 'legend':True if a==0 else None}
            # if col == 'np': kwgs['logy'] = True
            # if col == 'mixing_type': kwgs['kind'] = 'scatter'
            d.plot('q',col, label=cb, **kwgs)

            if col == 'temperature' and cb!='c0':
                d.plot('q','wimp_temp', label=f'{cb} DM', **kwgs)
                ax.semilogy()


    plt.show(block=False)


# fe----- plots -----#

###------------------ OLD from branch largeLH ------------------###
# fs----- plot luminosity profiles -----#
def plot_lums_profiles(pdf, hdf=None, title='', save=None):
    gb = pdf.groupby('profile_number')

    ncols = 2
    nrows = int(np.ceil(len(gb)/2))
    f, axs = plt.subplots(sharex=True, nrows=nrows,ncols=ncols, figsize=figsize)
    axs = axs.flatten()

    for i, (p, df) in enumerate(gb):
        ax = axs[i]

        # get luminosities in a dict
        df = df.sort_values('zone',ascending=False,axis=0) # sort to match dic items
        dic = eps_to_lum(df) # returns L_nuc, L_grav
        dic['L'] = df.luminosity
        dic.move_to_end('L',False)
        dic['extra_L'] = df.extra_L
        # plot
        for z, (key, L) in enumerate(dic.items()):
            if title=='c0' and key=='extra_L': continue
            ls = ':' if (key=='L' or key=='L_non-nuc-neu') else '-'
            ax.plot(df.q, L, ls=ls, zorder=-z, label=key)
            ax.axhline(0, c='0.5', lw=0.5)

        # add luminosities from hdf to check against integrated eps
        if hdf is not None:
            row = hdf.loc[hdf.profile_number==p,:]
            ax.scatter(1, 10**row.log_L, label=r'L')
            ax.scatter(1, 10**row.log_Lnuc, label=r'L$_{nuc}$')
            ax.scatter(1, 10**row.log_Lneu, label=r'L$_{\nu,tot}$')
            ax.scatter(1, row.eps_grav_integral, label=r'L$_{grav}$')
            if title!='c0':
                ax.scatter(1, row.extra_L, label=r'L$_{DM}$')

        if i==0: ax.legend(loc=10, ncol=2)
        ax.set_ylabel(r'luminosity [L$_\odot$]')
        ax.set_title(f'profile {p}')
    axs[-1].set_xlabel(r'q = m(<r)/M$_\star$')
    axs[-2].set_xlabel(r'q = m(<r)/M$_\star$')

    plt.suptitle(title)
    if save is not None: plt.savefig(save)
    plt.show(block=False)

    return dic

def eps_to_lum(df):
    df = df.sort_values('zone',ascending=False,axis=0) # sort star center -> sfc

    Mstar = df.iloc[-1].mass * Msun # grams

    dic = OD([])
    for eps in ['eps_nuc', 'non_nuc_neu', 'eps_grav']:#, 'eps_nuc_neu_total']:
        dqeps = df.dq * df[eps] # erg / g / s
        dqeps_sum = dqeps.cumsum() # integrate from center outward
        if eps == 'non_nuc_neu': eps = 'non-nuc-neu'
        dic['_'.join(['L', eps.split('_')[-1]])] = dqeps_sum * Mstar/Lsun # Lsun

    return dic

# fe----- plot luminosity profiles -----#

# fs----- plot luminosity histories -----#

# plot all luminosities
def plot_lums_history(lum_dict, profiles=None, hdf=None, title='', save=None):
    """ lum_dict (dict): should include age (x-axis) and luminosities
                         as returned by lums_dict() fnc above.
        profiles (list): 'all' or list of profile numbers to plot axvline
    """
    dic = lum_dict.copy()
    age = dic.pop('age')
    plt.figure(figsize=figsize)

    # plot luminosities
    for i, (sl, tup) in enumerate(dic.items()):
        l, ls = tup
        w = len(dic.keys())
        plt.plot(age,l,ls=ls,label=sl, zorder=w-i)
    plt.axhline(0, c='0.5', lw=0.5)

    # add vlines for models with profiles
    if profiles is not None:
        if profiles == 'all':
            vlines = hdf.loc[hdf.profile_number.notnull(), 'star_age']
            for v in vlines:
                if v < age.min(): continue
                plt.axvline(v, lw=0.5)
        else:
            vlines = hdf.loc[hdf.profile_number.isin(profiles), ['star_age','profile_number']]
            for index, row in vlines.iterrows():
                if row.star_age < age.min(): continue
                plt.axvline(row.star_age, lw=0.5)
                plt.annotate(f'{row.profile_number:.0f}',(row.star_age,0.5))

    plt.semilogx()
    plt.legend()
    plt.xlabel('star_age')
    plt.ylabel(r'luminosity [L$_\odot$]')
    plt.title(title)
    plt.tight_layout()
    if save is not None: plt.savefig(save)
    plt.show(block=False)

    return None

def plot_lums_history_06(lum_dict, save=None):
    """ lum_dict (dict): dict of dicts. Each individual dict
                         should include age (x-axis) and luminosities
                         as returned by lums_dict() fnc above.
    """
    plt.figure(figsize=figsize)

    for cb,d in lum_dict.items():
        dic = d.copy()
        age = dic.pop('age')

        # plot luminosities
        for i, (sl, tup) in enumerate(dic.items()):
            l, ls = tup
            w = len(dic.keys())
            plt.plot(age,l,ls=ls,label=rf'$\Gamma_B = {cb}$ ${sl}$', zorder=w-i)
    plt.axhline(0, c='0.5', lw=0.5)

    plt.semilogx()
    # plt.ylim(-1.1,5.0)
    plt.legend()
    plt.xlabel('star_age')
    plt.ylabel(r'luminosity [L$_\odot$]')
    plt.tight_layout()
    if save is not None: plt.savefig(save)
    plt.show(block=False)

    return None

# fe----- plot luminosities histories -----#

# fs----- check energy conservation -----#

def plot_energy_cons_error(hdf_dict, title='', save=None):
    """ Compare to Paxton 19 Figure 25, middle panel
    """
    plt.figure()

    for cb, hdf in hdf_dict.items():
        y = np.log10(np.abs(hdf.error_in_energy_conservation/hdf.total_energy))
        plt.plot(hdf.star_age, y, label=rf'$\Gamma_B = {cb}$')
    plt.axhline(0,c='0.5',lw=0.5)
    plt.semilogx()
    plt.xlabel('star age')
    plt.ylabel('log |Rel. Energy Error|')
    plt.legend()
    plt.title(title)
    if save is not None: plt.savefig(save)
    plt.show(block=False)

# plot luminosity excess
def plot_lum_excess(hdf_dict, title='', save=None):
    plt.figure()

    lums = ['age','L','Lnuc','Lgrav','Ltneu']
    for cb, hdf in hdf_dict.items():
        dic = lums_dict(hdf, lums)
        age = dic['age']
        L = dic['L'][0]
        Lnuc = dic['Lnuc'][0]
        Lgrav = dic['Lgrav'][0]
        Ltneu = dic['Ltneu'][0]
        plt.plot(age, L+Ltneu-Lnuc-Lgrav, label=rf'$\Gamma_B = {cb}$')

    plt.semilogx()
    plt.xlabel('star_age')
    plt.ylabel(r'L$_{excess}$ = L+L$_\nu$-L$_{nuc}$-L$_{grav}$ [L$_\odot$]')
    plt.legend()
    plt.title(title)
    plt.tight_layout()
    if save is not None: plt.savefig(save)
    plt.show(block=False)

    return None

def plot_energy(hdf, title='', save=None):
    plt.figure()
    cols = ['total_extra_heating','total_energy_sources_and_sinks','extra_energy']
    hdf.plot('star_age',cols, subplots=True)
    plt.semilogx()
    plt.title(title)
    if save is not None: plt.savefig(save)
    plt.show(block=False)

    return None

# fe----- check energy conservation -----#


# fs----- plot other history columns -----#
def plot_history(hdf,cols):
    nrows = len(cols)
    ncols = 1
    f, axs = plt.subplots(sharex=True, nrows=nrows,ncols=ncols, figsize=figsize)

    for i, (ax, col) in enumerate(zip(axs,cols)):
        for rk, df in hdf.groupby(level=['run_key','cb','mass']):
            c = rk[1]/6.
            ax.plot(df.star_age,df[col], label=rk)#, c=c)

            if col=='log_center_T' and c==1:
                ax.plot(df.star_age,np.log10(df.wimp_temp),ls=':')

        if i==0: ax.legend()
        ax.set_ylabel(col)

    ax.set_xlabel('star_age')
    plt.semilogx()
    plt.show(block=False)
    return None

def plot_burning_06(hdf_dict, title='', save=None):
    """
    """
    plt.figure()

    for cb, hdf in hdf_dict.items():
        plt.plot(hdf.star_age, hdf.pp, label=rf'$\Gamma_B = {cb}$, pp')
        plt.plot(hdf.star_age, hdf.cno, label=rf'$\Gamma_B = {cb}$, cno')
    plt.semilogx()
    plt.xlabel('star age')
    plt.ylabel(r'log burning [L$_\odot$]')
    plt.legend()
    plt.title(title)
    if save is not None: plt.savefig(save)
    plt.show(block=False)

def plot_center_abundances(hdf_dict, title='', save=None):
    plt.figure()

    for cb, hdf in hdf_dict.items():
        ls = ':' if cb=='10^6' else '-'
        # plt.plot(hdf.star_age, hdf.center_nx, ls=ls, label=rf'$\Gamma_B = {cb}$, nx$_c$')
        # plt.plot(hdf.star_age, hdf.center_np, ls=ls, label=rf'$\Gamma_B = {cb}$, np$_c$')
        plt.plot(hdf.star_age, hdf.center_h1, ls=ls, label=rf'$\Gamma_B = {cb}$, h1$_c$')
    plt.loglog()
    plt.xlabel('star age')
    plt.ylabel('fractional abundance')
    plt.legend()
    plt.title(title)
    plt.tight_layout()
    if save is not None: plt.savefig(save)
    plt.show(block=False)

def plot_debug(hdf_dict, title='', save=None):
    """ save (dict) with keys same as hdf_dict
    """
    cols = ['num_retries','num_backups',#'num_newton_iterations',
            'log_dt','model_number']

    for cb, hdf in hdf_dict.items():
        hdf.plot('star_age',cols, subplots=True, logx=True, grid=True)

        plt.xlabel('star age')
        plt.legend()
        plt.suptitle(rf'$\Gamma_B = {cb}$')
        if save is not None: plt.savefig(save[cb])
        plt.show(block=False)

# fe----- plot other history columns -----#

# fs----- plot other profiles -----#
def plot_nx_profiles(pdf, log=False, title='', save=None):
    gb = pdf.groupby('profile_number')

    ncols = 2
    nrows = int(np.ceil(len(gb)/2))
    f, axs = plt.subplots(sharex=True, nrows=nrows,ncols=ncols, figsize=figsize)
    axs = axs.flatten()

    for i, (p, df) in enumerate(gb):
        ax = axs[i]
        d = df.loc[df.q<1.5,:]
        ax.plot(d.q, d.nx)
        ax.plot(d.q, d.np)
        ax.axhline(0, c='k',lw=0.5)

        if log:
            ax.semilogy()
        ax.grid()
        if i==0: ax.legend()
        ax.set_ylabel(r'number density [cm$^{-3}$]')
        ax.set_title(f'profile {p}')
    axs[-1].set_xlabel(r'q = m(<r)/M$_\star$')
    axs[-2].set_xlabel(r'q = m(<r)/M$_\star$')

    plt.suptitle(title)
    if save is not None: plt.savefig(save)
    plt.show(block=False)

    return None

def plot_abundance_profiles(pdf, title='', save=None):
    gb = pdf.groupby('profile_number')

    ncols = 2
    nrows = int(np.ceil(len(gb)/2))
    f, axs = plt.subplots(sharex=True, nrows=nrows,ncols=ncols, figsize=figsize)
    axs = axs.flatten()

    for i, (p, df) in enumerate(gb):
        ax = axs[i]
        d = df.loc[df.q<0.5,:]
        ax.plot(d.q, d.x, label=r'H mass frac')
        ax.plot(d.q, d.y, label=r'He mass frac')

        ax.grid()
        if i==0: ax.legend(loc=10)
        ax.set_ylabel('mass fraction')
        ax.set_title(f'profile {p}')
    axs[-1].set_xlabel(r'q = m(<r)/M$_\star$')
    axs[-2].set_xlabel(r'q = m(<r)/M$_\star$')

    plt.suptitle(title)
    if save is not None: plt.savefig(save)
    plt.show(block=False)

    return None

def plot_eps_profiles(pdf, title='', save=None):
    gb = pdf.groupby('profile_number')

    ncols = 2
    nrows = int(np.ceil(len(gb)/2))
    f, axs = plt.subplots(sharex=True, nrows=nrows,ncols=ncols, figsize=figsize)
    axs = axs.flatten()

    for i, (p, df) in enumerate(gb):
        ax = axs[i]
        d = df.loc[df.q<0.5,:]
        ax.plot(d.q, d.eps_nuc, label=r'$\epsilon_{nuc}$')
        # ax.plot(d.q, d.eps_grav, label=r'$\epsilon_{grav}$')
        if title=='c6':
            ax.plot(d.q, d.extra_heat, label=r'$\epsilon_{DM}$')
        ax.axhline(0, c='0.5', lw=0.5)

        ax.grid()
        if i==0: ax.legend(loc=10)
        ax.set_ylabel('energy [erg/g/s]')
        ax.set_title(f'profile {p}')
    axs[-1].set_xlabel(r'q = m(<r)/M$_\star$')
    axs[-2].set_xlabel(r'q = m(<r)/M$_\star$')

    plt.suptitle(title)
    if save is not None: plt.savefig(save)
    plt.show(block=False)

    return None

def plot_rho_profiles(pdf, title='', save=None):
    gb = pdf.groupby('profile_number')

    ncols = 2
    nrows = int(np.ceil(len(gb)/2))
    f, axs = plt.subplots(sharex=True, nrows=nrows,ncols=ncols, figsize=figsize)
    axs = axs.flatten()

    for i, (p, df) in enumerate(gb):
        ax = axs[i]
        d = df.loc[df.q<1.15,:]
        ax.plot(d.q, d.logRho, label=r'Rho')

        if i==0: ax.legend(loc=10)
        ax.set_ylabel('log Rho')
        ax.set_title(f'profile {p}')
    axs[-1].set_xlabel(r'q = m(<r)/M$_\star$')
    axs[-2].set_xlabel(r'q = m(<r)/M$_\star$')

    plt.suptitle(title)
    if save is not None: plt.savefig(save)
    plt.show(block=False)

    return None

def plot_T_profiles(pdf, title='', save=None):
    gb = pdf.groupby('profile_number')

    ncols = 2
    nrows = int(np.ceil(len(gb)/2))
    f, axs = plt.subplots(sharex=True, nrows=nrows,ncols=ncols, figsize=figsize)
    axs = axs.flatten()

    for i, (p, df) in enumerate(gb):
        ax = axs[i]
        d = df.loc[df.q<0.15,:]
        ax.plot(d.q, d.logT, label=r'T')
        if title=='c6':
            ax.plot(d.q, np.log10(d.wimp_temp), label=r'T$_{DM}$')

        if i==0: ax.legend(loc=10)
        ax.set_ylabel('log temp [K]')
        ax.set_title(f'profile {p}')
    axs[-1].set_xlabel(r'q = m(<r)/M$_\star$')
    axs[-2].set_xlabel(r'q = m(<r)/M$_\star$')

    plt.suptitle(title)
    if save is not None: plt.savefig(save)
    plt.show(block=False)

    return None

def plot_Tx_minus_T_profiles(pdf, save=None):
    gb = pdf.groupby('profile_number')

    ncols = 2
    nrows = int(np.ceil(len(gb)/2))
    f, axs = plt.subplots(sharex=True, nrows=nrows,ncols=ncols, figsize=figsize)
    axs = axs.flatten()

    for i, (p, df) in enumerate(gb):
        ax = axs[i]
        d = df.loc[df.q<0.05,:]

        ax.plot(d.q, d.wimp_temp - 10**d.logT, label=r'T$_{DM}$ - T')
        ax.axhline(0, c='k', lw=0.5)

        if i==0: ax.legend(loc=10)
        ax.set_ylabel('temp [K]')
        ax.set_title(f'profile {p}')
    axs[-1].set_xlabel(r'q = m(<r)/M$_\star$')
    axs[-2].set_xlabel(r'q = m(<r)/M$_\star$')

    if save is not None: plt.savefig(save)
    plt.show(block=False)

    return None

def plot_convection(pdf, save=None):
    plt.figure()
    g = pdf.groupby('profile_number')
    g.plot('q','mixing_type', ax=plt.gca(), kind='scatter')
    if save is not None: plt.savefig(save)
    plt.show(block=False)


# fe----- plot other profiles -----#



# fs----- high masses -----#
def plot_col(hdf, col='log_L'):
    g = hdf.loc[hdf.star_age>1e7,:].groupby(['run_key','cb','mass'])
    plt.figure(figsize=figsize)
    ax = plt.gca()
    for i,d in g:
        print(i[1])
        d.plot('star_age',col, ax=ax, label=i, logx=True)
    plt.show(block=False)
    return None

def plot_mstau(hdf, title='', save=None):
    msdf = calc_MSlifetimes(hdf)

    plt.figure()
    ax = plt.gca()
    g = msdf.groupby(['run_key','cb'])
    for i,d in g:
        cb = i[1]
        d = d.reset_index('mass')
        plt.scatter(d.mass,d.deltaMS, label=cb, lw=cb, zorder=-cb)

    plt.ylim((-0.79,0.21))
    plt.semilogx()
    plt.xlabel(r'Star Mass [M$_\odot$]')
    plt.ylabel(r'$\Delta \tau_{MS} / \tau_{MS,\ NoDM}$')
    plt.legend()
    plt.title(title)
    plt.tight_layout()
    if save is not None: plt.savefig(save)
    plt.show(block=False)

def calc_MSlifetimes(hdf):
    """ hdf is DataFrame of multiple stars
    """
    msdf = pd.DataFrame(index=hdf.index.unique(), columns=['MStau', 'deltaMS'])
    g = hdf.groupby(['run_key','cb','mass'])
    for i,d in g:
        msdf.loc[i,'MStau'], __,__ = get_MStau(d)
        if i[1] == 0:
            msdf.loc[i,'deltaMS'] = 0
        else:
            ms0 = msdf.loc[(i[0],0,i[2]),'MStau']
            msdf.loc[i,'deltaMS'] = (msdf.loc[i,'MStau'] - ms0)/ms0

    return msdf



# fe----- high masses -----#
