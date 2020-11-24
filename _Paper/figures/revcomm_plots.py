import pandas as pd
import plot_fncs as pf
from matplotlib import pyplot as plt

kerg = 1.3806504e-16 # [erg K^-1]
standard_cgrav = 6.67428e-8 # [g^-1 cm^3 s^-2]
msol = 1.9892e33 # [g]
rsol = 6.9598e10 # [cm]
gperGeV = 1.78266e-24 # grams to GeV/c^2 conversion

def plot_evaporation_mass(cb=0, masses=[], save=None):

    hdfs = []
    for mass in masses:
        hdf = pf.load_hist_from_file(cb, mass=mass)
        hdf['mass'], hdf['cb'] = mass, cb
        hdf.sort_values('star_age', inplace=True)
        h = hdf.loc[hdf.center_h1<0.5,:].iloc[0]
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
    plt.title(r'm$_{\chi,evap}$ = (2k$_B$T$_c$) / (3GM$_\star$/R$_\star$) (values taken at X$_c$ = 0.5)')
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
