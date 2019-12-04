# paths assume this is being run on Roy
import os
from collections import OrderedDict as OD
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

mesaruns = '/Users/troyraen/Osiris/DMS/mesaruns'
dr = mesaruns + '/RUNS_largeLH'
c6path = dr+ '/c6/m1p0/LOGS'
hpath = c6path+ '/history.data'
pipath = c6path+ '/profiles.index'

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
hdf = pd.read_csv(hpath, header=4, sep='\s+').set_index('model_number', drop=False)
pidx_cols = ['model_number', 'priority', 'profile_number']
pidf = pd.read_csv(pipath, names=pidx_cols, skiprows=1, header=None, sep='\s+')
# add profile numbers to hdf
hdf['profile_number'] = pidf.set_index('model_number').profile_number

# fe----- load data -----#


# fs----- plot luminosity profiles -----#
def plot_lums_profiles(pdf):
    gb = pdf.groupby('profile_number')

    ncols = 2
    nrows = int(np.ceil(len(gb)/2))
    f, axs = plt.subplots(sharex=True, nrows=nrows,ncols=ncols, figsize=(14,8))
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
        for key, L in dic.items():
            ls = ':' if key == 'L' else '-'
            ax.plot(df.q, L, ls=ls, label=key)
            ax.axhline(0, c='0.5', lw=0.5)

        if i==0: ax.legend(loc=10)
        ax.set_ylabel(r'luminosity [L$_\odot$]')
        ax.set_title(f'profile {p}')
    axs[-1].set_xlabel(r'q = m(<r)/M$_\star$')
    axs[-2].set_xlabel(r'q = m(<r)/M$_\star$')

    plt.show(block=False)

    return dic


def eps_to_lum(df):
    df = df.sort_values('zone',ascending=False,axis=0) # sort star center -> sfc

    Lsun = 3.8418e33 # erg/s
    Msun = 1.9892e33 # grams
    Mstar = df.iloc[-1].mass * Msun # grams

    dic = OD([])
    for eps in ['eps_nuc', 'eps_grav']:#, 'eps_nuc_neu_total']:
        dqeps = df.dq * df[eps] # erg / g / s
        dqeps_sum = dqeps.cumsum() # integrate from center outward
        dic['_'.join(['L', eps.split('_')[-1]])] = dqeps_sum * Mstar/Lsun # Lsun

    return dic

# fe----- plot luminosity profiles -----#



# fs----- plot luminosities v age -----#

# plot all luminosities
def plot_lums_history(lum_dict, profiles=None):
    """ lum_dict (dict): should include age (x-axis) and luminosities
                         see dic defined in main.py for structure.
        profiles (list): 'all' or list of profile numbers to plot axvline
    """
    dic = lum_dict.copy()
    age = dic.pop('age')
    plt.figure(figsize=(14,8))

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
                plt.annotate(f'{row.profile_number:.0f}',(row.star_age,3))

    plt.semilogx()
    plt.legend()
    plt.xlabel('star_age')
    plt.ylabel(r'luminosity [L$_\odot$]')
    plt.tight_layout()
    plt.show(block=False)

    return None


# plot excess
def plot_lum_excess(age,L,Lnuc,Lgrav):
    plt.figure()
    plt.plot(age, L-Lnuc-Lgrav)
    plt.semilogx()
    plt.xlabel('star_age')
    plt.ylabel(r'L-L$_{nuc}$-L$_{grav}$ [L$_\odot$]')
    plt.tight_layout()
    plt.show(block=False)


# fe----- plot luminosities v age -----#
