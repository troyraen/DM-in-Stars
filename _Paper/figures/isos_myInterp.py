from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

from data_proc import iter_starModel_dirs, load_history
from plot_fncs import datadir, isomy_csv

usepruned = True

def write_isomy_csv(isodf):
    try:
        isodf.to_csv(isomy_csv)
    except:
        raise
    else:
        print(f'\nsuccessfully wrote {isomy_csv}\n')

    return

def interp4isos(test=False):
    """
    Args
        test = (mass, cb) to return this model's data + interpolation results
                False to run normally.
    """

    idflist = []

    for massdir, mass, cb in iter_starModel_dirs(Path(datadir)):
        # if test == (some mass, cb), skip everything else
        if test not in [False, (mass, cb)]: continue

        # load history file
        fin = 'history_pruned.data' if usepruned else 'history.data'
        hdf = load_history(massdir/f'LOGS/{fin}')#, (mass,cb))

        # interpolate columns to isoAges
        print(f'interpolating {mass} c{cb}')
        idf, isoAges, interpCols = interp_hdf(hdf)

        # add static columns
        idf['mass'], idf['cb'] = mass, cb

        if test == (mass, cb):
            return (idf, isoAges, interpCols)

        idflist.append(idf.loc[idf.index.isin(isoAges),:])
        # if len(idflist)>2: break

    # concat and cleanup
    isodf = pd.concat(idflist)
    isodf.rename(columns={'mass':'initial_mass', 'cb':'cboost'}, inplace=True)
    isodf.sort_values(['initial_mass','cboost'], inplace=True)

    write_isomy_csv(isodf)

    return isodf

def interp_hdf(hdf):
    """ interpolates hdf to isoAges
    """
    isoAges = pd.Series(np.logspace(7.5, 10.2, num=100)).apply(np.log10)
    interpCols = ['log_L', 'log_Teff', 'center_h1', 'center_he4']

    # prep the hdf
    hdf['log10_star_age_yr'] = hdf.star_age.apply(np.log10)
    l = len(hdf)
    hdf.drop_duplicates(subset='log10_star_age_yr', inplace=True)
    d = l - len(hdf)
    if d != 0:
        print(f'\t*** dropped {d} duplicates from hdf\n')
    maxage = hdf.log10_star_age_yr.max()

    # create new df and do the interpolation
    allAges = pd.concat([isoAges, hdf.log10_star_age_yr], axis=0, ignore_index=True) # series
    data = {col: hdf.set_index('log10_star_age_yr')[col] for col in interpCols}
    df = pd.DataFrame(index=allAges, data=data) # NaNs for rows not in hdf
    df.index.rename('log10_isochrone_age_yr', inplace=True)
    df.sort_index(inplace=True)
    # cut isoAges > max star_age
    df = df.loc[df.index<=maxage,:]
    # interpolate columns to isoAges
    df.interpolate(method='index', inplace=True)

    return (df, isoAges, interpCols)

def plot_test_interp_hdf(mass, cb, save=None):
    """ Plots interpolated columns of the (mass, cb) hdf.
    """
    # get the right df and add some columns
    df, isoAges, interpCols = interp4isos(test=(mass,cb))
    df = df.loc[df.index>isoAges.min(),:]
    df = df.reset_index().set_index('log10_isochrone_age_yr', drop=False)
    df['is_isoAge'] = df.log10_isochrone_age_yr.isin(isoAges)
    df.replace({True:1, False:0}, inplace=True)
    df['ms'] = df.is_isoAge*25+25

    # plot
    title = f'{mass} Msun c{cb}'
    plot_interpCols(df, interpCols, title=title, save=save)

    # zoom in and plot again
    tmp = save.rsplit('.',1)
    s = '.'.join([tmp[0]+'_zoom','png'])
    y, o = df.log10_isochrone_age_yr.min(), df.log10_isochrone_age_yr.max()
    zoomage = o - (o-y)/5
    d = df.loc[df.log10_isochrone_age_yr>zoomage,:]
    plot_interpCols(d, interpCols, title=title, save=s)

    return

def plot_interpCols(df, interpCols, title=None, save=None):
    fig, axs = plt.subplots(nrows=len(interpCols))
    fig.suptitle(title)

    for a, col in enumerate(interpCols):
        kwargs = {  'x':'log10_isochrone_age_yr', 'y':col,
                    'ax':axs[a], 's':df.ms, 'alpha':0.5,
                    'c':'is_isoAge', 'colormap':'coolwarm',
                }
        df.plot.scatter(**kwargs)

    if save is not None: plt.savefig(save)
    plt.show(block=False)
