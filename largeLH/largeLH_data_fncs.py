# paths assume this is being run on Roy

# fs----- imports, paths, constants -----#
import os
from collections import OrderedDict as OD
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

mesaruns = '/Users/troyraen/Osiris/DMS/mesaruns'
dr_r10398 = mesaruns + '/RUNS_largeLH'
dr_r12115 = mesaruns + '/RUNS_largeLH_mesa-r12115' # using newest MESA release r12115

dr = dr_r12115
c6path = dr+ '/c6/m1p0/LOGS'
hpath = c6path+ '/history.data'
pipath = c6path+ '/profiles.index'

c0path = dr+ '/c0/m1p0/LOGS'
h0path = c0path+ '/history.data'
pi0path = c0path+ '/profiles.index'

Lsun = 3.8418e33 # erg/s
Msun = 1.9892e33 # grams

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
def load_main_data():
    hdf = pd.read_csv(hpath, header=4, sep='\s+').set_index('model_number', drop=False)
    pidx_cols = ['model_number', 'priority', 'profile_number']
    pidf = pd.read_csv(pipath, names=pidx_cols, skiprows=1, header=None, sep='\s+')
    # add profile numbers to hdf
    hdf['profile_number'] = pidf.set_index('model_number').profile_number

    h0df = pd.read_csv(h0path, header=4, sep='\s+').set_index('model_number', drop=False)
    pi0df = pd.read_csv(pi0path, names=pidx_cols, skiprows=1, header=None, sep='\s+')
    h0df['profile_number'] = pi0df.set_index('model_number').profile_number

    return (hdf, pidf, h0df, pi0df)

def load_profiles_df(pnums4df, cb=6):
    dfs = []
    for p in pnums4df:
        if cb==6: path = c6path
        elif cb==0: path = c0path
        ppath = path+ f'/profile{p}.data'
        pdf = pd.read_csv(ppath, header=4, sep='\s+')
        pdf['profile_number'] = p
        dfs.append(pdf.set_index(['profile_number','zone']))
    pdf = pd.concat(dfs, axis=0)

    return pdf

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



# fe----- load data -----#


# fs----- plot luminosity profiles -----#
def plot_lums_profiles(pdf, hdf=None, title=''):
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
def plot_lums_history(lum_dict, profiles=None, hdf=None, title=''):
    """ lum_dict (dict): should include age (x-axis) and luminosities
                         as returned by lums_dict() fnc above.
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
                plt.annotate(f'{row.profile_number:.0f}',(row.star_age,0.5))

    plt.semilogx()
    plt.legend()
    plt.xlabel('star_age')
    plt.ylabel(r'luminosity [L$_\odot$]')
    plt.title(title)
    plt.tight_layout()
    plt.show(block=False)

    return None

def plot_lums_history_06(lum_dict):
    """ lum_dict (dict): dict of dicts. Each individual dict
                         should include age (x-axis) and luminosities
                         as returned by lums_dict() fnc above.
    """
    plt.figure(figsize=(14,8))

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
    plt.legend()
    plt.xlabel('star_age')
    plt.ylabel(r'luminosity [L$_\odot$]')
    plt.tight_layout()
    plt.show(block=False)

    return None

# fe----- plot luminosities v age -----#

# fs----- check energy conservation -----#

def plot_energy_cons_error(hdf_dict, title=''):
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
    plt.show(block=False)

# plot luminosity excess
def plot_lum_excess(hdf_dict):
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
    plt.tight_layout()
    plt.show(block=False)

    return None


# def plot_lum_excess(age,L,Lnuc,Lgrav,Ltneu):
#     plt.figure()
#     plt.plot(age, L+Ltneu-Lnuc-Lgrav)
#     plt.semilogx()
#     plt.xlabel('star_age')
#     plt.ylabel(r'L+L$_{\nu,therm}$-L$_{nuc}$-L$_{grav}$ [L$_\odot$]')
#     plt.tight_layout()
#     plt.show(block=False)


def plot_energy(hdf, title=''):
    plt.figure()
    cols = ['total_extra_heating','total_energy_sources_and_sinks','extra_energy']
    hdf.plot('star_age',cols, subplots=True)
    plt.semilogx()
    plt.title(title)
    plt.show(block=False)

    return None

# fe----- check energy conservation -----#


# fs----- plot other history columns -----#
def plot_burning_06(hdf_dict, title=''):
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
    plt.show(block=False)

def plot_center_abundances(hdf_dict, title=''):
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
    plt.show(block=False)


# fe----- plot other history columns -----#

# fs----- plot other profiles -----#
def plot_nx_profiles(pdf, log=False, title=''):
    gb = pdf.groupby('profile_number')

    ncols = 2
    nrows = int(np.ceil(len(gb)/2))
    f, axs = plt.subplots(sharex=True, nrows=nrows,ncols=ncols, figsize=(14,8))
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
    plt.show(block=False)

    return None

def plot_abundance_profiles(pdf, title=''):
    gb = pdf.groupby('profile_number')

    ncols = 2
    nrows = int(np.ceil(len(gb)/2))
    f, axs = plt.subplots(sharex=True, nrows=nrows,ncols=ncols, figsize=(14,8))
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
    plt.show(block=False)

    return None

def plot_eps_profiles(pdf, title=''):
    gb = pdf.groupby('profile_number')

    ncols = 2
    nrows = int(np.ceil(len(gb)/2))
    f, axs = plt.subplots(sharex=True, nrows=nrows,ncols=ncols, figsize=(14,8))
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
    plt.show(block=False)

    return None

def plot_rho_profiles(pdf, title=''):
    gb = pdf.groupby('profile_number')

    ncols = 2
    nrows = int(np.ceil(len(gb)/2))
    f, axs = plt.subplots(sharex=True, nrows=nrows,ncols=ncols, figsize=(14,8))
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
    plt.show(block=False)

    return None

def plot_T_profiles(pdf, title=''):
    gb = pdf.groupby('profile_number')

    ncols = 2
    nrows = int(np.ceil(len(gb)/2))
    f, axs = plt.subplots(sharex=True, nrows=nrows,ncols=ncols, figsize=(14,8))
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
    plt.show(block=False)

    return None

def plot_Tx_minus_T_profiles(pdf):
    gb = pdf.groupby('profile_number')

    ncols = 2
    nrows = int(np.ceil(len(gb)/2))
    f, axs = plt.subplots(sharex=True, nrows=nrows,ncols=ncols, figsize=(14,8))
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

    plt.show(block=False)

    return None

def plot_convection(pdf):
    plt.figure()
    g = pdf.groupby('profile_number')
    g.plot('q','mixing_type', ax=plt.gca(), kind='scatter')
    plt.show(block=False)


# fe----- plot other profiles -----#
