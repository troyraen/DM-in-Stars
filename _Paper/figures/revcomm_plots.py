import numpy as np
import pandas as pd
import plot_fncs as pf
from matplotlib import pyplot as plt

kerg = 1.3806504e-16 # [erg K^-1]
standard_cgrav = 6.67428e-8 # [g^-1 cm^3 s^-2]
msol = 1.9892e33 # [g]
rsol = 6.9598e10 # [cm]
gperGeV = 1.78266e-24 # grams to GeV/c^2 conversion

def plot_temps_v_Mstar(cb=0, masses=[], save=None):
    Xclim = 0.5
    hdfs = []
    for mass in masses:
        hdf = pf.load_hist_from_file(cb, mass=mass)
        if hdf is None: continue  # no model dir with these params
        hdf['mass'], hdf['cb'] = mass, cb
        hdf.sort_values('star_age', inplace=True)
        h = hdf.loc[hdf.center_h1<Xclim,:].iloc[0]
        print(f'X_c = {h.center_h1}')
        print(f'T_c = {h.center_T}')
        print(f'T_c = {h.wimp_temp}')
        print(f'M = {h.star_mass}')
        print()
        # hdf.set_index(['mass','cb'], inplace=True)
        hdfs.append(h)
    hdf = pd.concat(hdfs, axis=1, keys=[s.name for s in hdfs]).T

    plt.figure()
    plt.plot(hdf.mass, np.log10(hdf.center_T), label=r'T$_c$')
    plt.plot(hdf.mass, np.log10(hdf.wimp_temp), label=r'T$_{DM}$')

    for m, c in zip([1.00, 1.30], ['m','c']):
        print(f'hline for {m}')
        Tc = hdf.loc[hdf.mass==m,'center_T'].iloc[0]
        Tx = hdf.loc[hdf.mass==m,'wimp_temp'].iloc[0]
        lbl = f'M={m}, Tc={Tc:.5e}, TDM={Tx:.5e}'
        plt.axvline(m, c=c, label=lbl)

    plt.xlabel('M$_\star$')
    # plt.ylabel(r'(k T$_{c}$) / (GM/R)')
    plt.ylabel('log(T [K])')
    plt.legend()
    plt.title(f'Temperatures for cb{cb} models (values taken at ' + r'X$_c$ = ' + f'{Xclim})')
    plt.tight_layout()
    plt.savefig(save)

def plot_evaporation_mass(cb=0, masses=[], save=None):
    Xclim = 0.5
    hdfs = []
    for mass in masses:
        hdf = pf.load_hist_from_file(cb, mass=mass)
        hdf['mass'], hdf['cb'] = mass, cb
        hdf.sort_values('star_age', inplace=True)
        h = hdf.loc[hdf.center_h1<Xclim,:].iloc[0]
        print(f'X_c = {h.center_h1}')
        print(f'T_c = {h.center_T}')
        print(f'M = {h.star_mass}')
        print(f'R = {10**h.log_R}')
        print()
        # hdf.set_index(['mass','cb'], inplace=True)
        hdfs.append(h)
    hdf = pd.concat(hdfs, axis=1, keys=[s.name for s in hdfs]).T

    plt.figure()
    # y = (kerg * hdf.center_T) / (standard_cgrav * hdf.star_mass*msol / 10**hdf.log_R*rsol)
    mx = (2 * 10**hdf.log_R*rsol * kerg * hdf.center_T) / (3 * standard_cgrav * hdf.star_mass*msol) / gperGeV # GeV
    plt.plot(hdf.mass, mx)

    # mx = 2 R k Tc / 3 G M

    plt.xlabel('M$_\star$')
    # plt.ylabel(r'(k T$_{c}$) / (GM/R)')
    plt.ylabel(r'm$_{\chi,evap}$ [GeV]')
    note = f'{Xclim})'
    plt.title(r'm$_{\chi,evap}$ = $\frac{2k_B T_c}{3GM_\star/R_\star}$ (values taken at X$_c$ = '+note)
    plt.tight_layout()
    plt.savefig(save)

def plot_center_density(cbs=[], mass=1.0, save=None):

    plt.figure()

    for cb in cbs:
        hdf = pf.load_hist_from_file(cb, mass=mass)
        hdf = hdf.loc[hdf.star_age>1e7,:]
        cmapdict = pf.get_cmapdict(cb,len(hdf.star_age))
        plt.scatter(hdf.star_age, hdf.center_Rho, **cmapdict, s=1)
        c = pf.get_cmap_color(cb)
        plt.plot(hdf.star_age, hdf.center_Rho, c=c)#, label=f'gamma_B = {cb}')

    plt.loglog()
    cbar = pf.get_cbcbar()

    plt.xlabel('star age [yrs]')
    plt.ylabel('center density')
    plt.title(f'mass = {mass} Msun')
    plt.tight_layout()
    plt.savefig(save)
