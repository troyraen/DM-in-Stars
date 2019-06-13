# import plot_fncs as pf
# fs imports
import numpy as np
import pandas as pd
from pandas import DataFrame as DF
from collections import OrderedDict as OD
import math
from scipy.interpolate import interp1d as interp1d
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.colors import ListedColormap
from matplotlib.ticker import FormatStrFormatter
import os
# fe imports

# fs Set plot defaults:
# mpl.rcParams.update(mpl.rcParamsDefault)
print('Setting plot defaults.')
mpl.rcParams['figure.figsize'] = [8.0, 5.0]
savefigw = 11.2
savefigh = 7.
mpl.rcParams['font.size'] = 13
mpl.rcParams['lines.linewidth'] = 7
# mpl.rcParams['legend.fontsize'] = 'small'
plt.rcParams["text.usetex"] = False
# mpl.rcParams['axes.linewidth'] = 0.8
mpl.rcParams['axes.titlesize'] = 20
mpl.rcParams['axes.titleweight'] = 'normal'
mpl.rcParams['axes.labelsize'] = 28
# mpl.rcParams['axes.labelweight'] = 'heavy'
mpl.rcParams['xtick.labelsize'] = 20
mpl.rcParams['xtick.direction'] = 'inout'
mpl.rcParams['xtick.major.size'] = 6.5
mpl.rcParams['xtick.minor.visible'] = True
mpl.rcParams['xtick.minor.size'] = 4.0
mpl.rcParams['ytick.labelsize'] = 20
mpl.rcParams['ytick.direction'] = 'inout'
mpl.rcParams['ytick.major.size'] = 6.5
mpl.rcParams['ytick.minor.visible'] = True
mpl.rcParams['ytick.minor.size'] = 4.0

### Annotations
# This is probably the same for all plots
ann_fs = 26 # font size
ann_rmarg = 0.73 # use for right align
ann_lmarg = 0.05 # use for left align
ann_tmarg = 0.92 # top margin

# fe Set plot defaults

# fs Files and directories
mesaruns = '/Users/troyraen/Osiris/mesaruns'
datadir = mesaruns+ '/RUNS_2test_final/plotsdata/Glue'
plotdir = '/Users/troyraen/Documents/Comms/WIMP_Paper'
# finalplotdir = '/Users/troyraen/Osiris/mesaruns/RUNS_2test_final/plots'
# finalplotdir = '/Users/troyraen/Google_Drive/MESA/code/mesa_wimps/final_plots'
finalplotdir = '/Users/troyraen/Google_Drive/MESA/code/mesa_wimps/DMS-Paper/plots'
fzams = datadir+ '/zams.csv'
fdesc = datadir+ '/descDF_MS_allmassesreg.csv'
profiles_datadir = mesaruns+ '/RUNS_2test_final/profile_runs/'
iso_csv = mesaruns+ '/RUNS_2test_final/isochrones/isochrones.csv'
r2tf_dir = mesaruns+ '/RUNS_2test_final'

try: # mount Osiris dir if not already
    assert os.path.isdir(mesaruns)
    print('Osiris dir is already mounted.')
except:
    try:
        print('Mounting Osiris.')
        os.system("sshfs tjr63@osiris-inode01.phyast.pitt.edu:/home/tjr63 /Users/troyraen/Osiris/")
        assert os.path.isdir(mesaruns)
    except:
        print('Osiris remote mount failed.')
        print('Make sure Pulse Secure is connected and Korriban is not already mounted.')
        print('If Osiris is mounted, check that this path is valid: {}.'.format(mesaruns))
        raise

# fe Files and directories

# fs General helper fncs
def fix_desc_mass(descdf):
    descdf['mass'] = descdf.apply(get_mass_from_idx, axis=1)
    return descdf

def get_mass_from_idx(row):
    #     get star id
    idx = int(row['star_index'])

    #     parse to get mass
    mint = str(idx)[:-3]
    if int(mint[0])>5 or len(mint)==1:
        mass = float('0.'+mint)
    else:
        mass = float(mint[0]+'.'+mint[1:])
    return mass

def get_h1_modnums(frm='profile_metadata', mass=3.5, hdf=None, cb=None):
    """ Returns dict: key = cb, val = list of model numbers corresponding to
                                        center_h1 cuts in dict h1cuts = get_h1cuts()
        frm = 'profile_metadata' uses mdf for look up
        frm = 'history_data' uses a history df (must pass hdf and cb)
    """
    h1cuts = get_h1cuts()
    cbmods = {} # dict of lists of model numbers

    if frm == 'history_data':
        mods = []
        for name, h1c in h1cuts.items():
            try:
                # Get the first model number after center_h1 < h1c:
                tmp = hdf[hdf.center_h1<h1c].sort_values('model_number').model_number
                mods.append(tmp.iloc[0])
            except IndexError:
                print()
                print(name, 'no model with center_h1 < ', h1c)
                print('setting modnum to NaN')
                mods.append(float('NaN'))
            except:
                print()
                print('Make sure you passed a valid hdf to get_h1_modnums.')
        cbmods[cb] = mods


    elif frm == 'profile_metadata':
        cbg = mdf[mdf.initial_mass==mass].groupby('cb', sort=True) # ensure sorted by cb value
        for i, (cb, df) in enumerate(cbg):
            mods = [] # list of model numbers
            for name, h1c in h1cuts.items():
                try:
                    # Get the first model number after center_h1 < h1c:
                    tmp = df[df.center_h1<h1c].sort_values('model_number').model_number
                    mods.append(tmp.iloc[0])
                except IndexError:
                    print()
                    print(cb, name, 'no model with center_h1 < ', h1c)
                    print('setting modnum to NaN')
                    mods.append(float('NaN'))

            cbmods[cb] = mods
    else:
        print('Must pass valid argument with frm. (profile_metadata or history_data)')

    return cbmods

def get_h1cuts():
    """ Returns OD of h1 cuts to find profile model numbers"""
    start_center_h1 = 0.718 # this is the same for all models

    # define h1 cuts to find profile model numbers
    # taken from Dotter where possible
    h1cuts = OD([('ZAMS',start_center_h1 - 0.0015),
                 ('IAMS',0.3),
                 ('H-1',1e-1),
                 ('H-2',1e-2),
                 ('H-3',1e-3),
                 ('H-4',1e-4),
                 ('H-6',1e-6),
                 ('TAMS',1e-12)])

    return h1cuts

def cut_HR_hdf(hdf):
    h1cuts = get_h1cuts()
    ZAMS_cut = h1cuts['ZAMS']
    df = hdf[hdf.center_h1 < ZAMS_cut]
    return df


# fe General helper fncs

# fs Colormap helper fncs
# Colormap and colorbar functions common to all color maps

# used in plt.plot and other places to normalize colorbar
cbvmin=-0.5
cbvmax=6.5

def normalize_RGB(rgb_tuple):
    """ rgb_tuple assumed to be of length 3, values in range [0,255]
        Returns tuple with rgb_tuple values divided by 255,
            with 1 appended to end of tuple (needed for alpha in colormaps)
    """

    rbgnorm = 255.0
    rgb_scaled = [ c/rbgnorm for c in rgb_tuple ]
    rgb_scaled.append(1.0)
    return tuple(rgb_scaled)

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    """ Returns a new color map, trucated at min/maxval.
        Send minval and maxval as lists to slice out the middle of the colorbar
            e.g. minval = [0.0,0.55], maxval = [0.45, 1.0] removes the middle 10%.
    """
    if type(minval)==list:
        nn = int(n/len(minval))
        vals_lst = []
        for i in range(len(minval)):
            vals_lst.append(np.linspace(minval[i], maxval[i], nn))
        sample_vals = np.concatenate(vals_lst)
        mnval = minval[0]; mxval = maxval[-1]
    else:
        sample_vals = np.linspace(minval, maxval, n)
        mnval = minval; mxval = maxval

    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=mnval, b=mxval),
        cmap(sample_vals))

    return new_cmap

def get_cmap_color(val, cmap=None, myvmin=cbvmin, myvmax=cbvmax):
    """ Returns rgba color of val in cmap normalized to global vmin, vmax"""
    if cmap==None:
        cmap = cbcmap
    norm = mpl.colors.Normalize(vmin=myvmin, vmax=myvmax)
    rgba = cmap(norm(val))
    return rgba

def get_cmapdict(pos, l, which_cmap='cbcmap'):
    """ pos = position in colormap
            = cb for cbcmap
            = cmap_masses[mass] for mcmap

        l = length of plotted array.

        which_cmap = 'cbcmap' (for cb) or 'mcmap' (for mass)

        Returns a dict to be used as kwargs in plt.plot.
        all get the same color (specific cb or mass)
    """

    if which_cmap=='cbcmap':
        cmp = cbcmap
        vmn = cbvmin
        vmx = cbvmax
    elif which_cmap=='mcmap':
        cmp = mcmap
        vmn = mvmin
        vmx = mvmax
    else:
        print('Invalid argument passed to which_cmap in get_cmapdict()')

    if l == 1:
        return {'c':pos, 'cmap':cmp, 'vmin':vmn, 'vmax':vmx}
    else:
        return {'c':pos*np.ones(l), 'cmap':cmp, 'vmin':vmn, 'vmax':vmx}
# fe Colormap helper fncs

# fs cboost colormap
def get_cbcbar(sm=None, cax=None, f=None, **kwargs):
    """ Returns a colorbar that will be added to the side of a plot.
        Intended to be used with plotting that uses cbcmap as the colormap.

        Pass cax = subplot axis if using on figure with subplots
        Example:
                f.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.8, \
                                                    wspace=0.02, hspace=0.02)
                cb_ax = f.add_axes([0.83, 0.1, 0.02, 0.8])
                cbar = get_cbcbar(sm=ascat, cax=cb_ax, f=f)

    """
    cbar = plt.colorbar(sm, **{'pad':0.01}, **kwargs) if cax is None \
            else f.colorbar(sm, cax=cax, **kwargs)
    cbar.set_ticks([0,1,2,3,4,5,6],update_ticks=False)
    cbar.set_ticklabels(['NoDM',r'$10^1$',r'$10^2$',r'$10^3$',r'$10^4$',r'$10^5$',r'$10^6$'])
    cbar.ax.minorticks_off()
    cbar.set_label(r'$\Gamma_B$', labelpad=-5, rotation='horizontal')
    return cbar
# YlGnBu
rbgnorm = 255.0
c0c = normalize_RGB((199,233,180))
c1c = normalize_RGB((127,205,187))
c2c = normalize_RGB((65,182,196))
c3c = normalize_RGB((29,145,192))
c4c = normalize_RGB((34,94,168))
c5c = normalize_RGB((37,52,148))
c6c = normalize_RGB((8,29,88))
carr = (c0c,c1c,c2c,c3c,c4c,c5c,c6c)
cbcmap = ListedColormap(carr)

# fe cboost colormap

# fs mass colormap
m5p0c = normalize_RGB((254,227,145))
m3p5c = normalize_RGB((254,196,79))
m2p0c = normalize_RGB((254,153,41))
m1p0c = normalize_RGB((217,95,14))
m0p8c = normalize_RGB((153,52,4))
marr = (m0p8c,m1p0c,m2p0c,m3p5c,m5p0c)
mcmap = ListedColormap(marr)

# These need to match above. Used to generate mcbar (below)
cmap_masses = OD([(0.8,0), (1.0,1), (2.0,2), (3.5,3), (5.0,4)])
    # key = mass
    # val = position in colormap. e.g.: get_cmap_color(cmap_masses[mass], cmap=mcmap, myvmin=mvmin, myvmax=mvmax)

# used in plt.plot and other places to normalize colorbar:
mvmin = -0.5
mvmax = len(cmap_masses) - 0.5

def get_mcbar(sm=None, cax=None, f=None, **kwargs):
    """ Returns a colorbar that will be added to the side of a plot.
        Intended to be used with plotting that uses cbcmap as the colormap.

        Pass cax = subplot axis if using on figure with subplots
        Example:
                f.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.8, \
                                                    wspace=0.02, hspace=0.02)
                cb_ax = f.add_axes([0.83, 0.1, 0.02, 0.8])
                cbar = get_mcbar(sm=ascat, cax=cb_ax, f=f)

    """
    cbar = plt.colorbar(sm, **{'pad':0.01}, **kwargs) if cax is None \
            else f.colorbar(sm, cax=cax, **kwargs)
    cbar.set_ticks([i for i in range(len(cmap_masses.keys()))],update_ticks=False)
    cbar.set_ticklabels([m for m in cmap_masses.keys()])
    cbar.ax.minorticks_off()
    cbar.set_label(r'Star Mass [M$_{\odot}$]', labelpad=5)
    return cbar

# fe mass colormap

# fs isochrone colormap
# Create iso colormap (isocmap):
# cmap_BuRd = plt.get_cmap('RdBu_r')
# isocmap = truncate_colormap(cmap_BuRd, minval=[0.0,0.52], maxval=[0.48, 1.0], n=500)
cmap_BuPu = plt.get_cmap('BuPu')
isocmap = truncate_colormap(cmap_BuPu, minval=0.25, maxval=1.0, n=500)

# used in plt.plot and other places to normalize colorbar:
isovmin = 8.0
isovmax = 10.25

def get_isocbar(sm=None, cax=None, f=None, **kwargs):
    """ Returns a colorbar that will be added to the side of a plot.
        Intended to be used with plotting that uses cbcmap as the colormap.

        Pass cax = subplot axis if using on figure with subplots
        Example:
                f.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.8, \
                                                    wspace=0.02, hspace=0.02)
                cb_ax = f.add_axes([0.83, 0.1, 0.02, 0.8])
                cbar = get_mcbar(sm=ascat, cax=cb_ax, f=f)

    """
    cbar = plt.colorbar(sm, **{'pad':0.01}, **kwargs) if cax is None \
            else f.colorbar(sm, cax=cax, **kwargs)
#     cbar.set_ticks([i for i in range(len(cmap_masses.keys()))],update_ticks=False)
#     cbar.set_ticklabels([m for m in cmap_masses.keys()])
    cbar.ax.minorticks_off()
    cbar.set_label(r'log(Isochrone Age /yrs)', labelpad=5)
    return cbar
# fe isochrone colormap

# fs Profiles setup
def get_pidx(cb, modnum, mass=3.5):
    return 'm{mass}c{cb}mod{modnum}'.format(mass=mass, cb=cb, modnum=modnum)

def get_profile_path(row):
    """ row should be a row from df pidxdf.
        Returns path to the profile#.data file"""
    path = row.path_to_LOGS
    pnum = row.profile_number
    fprof = path+ '/profile'+ str(pnum)+ '.data'
    return fprof

def get_profilesindex_df(profile_runs_dir=profiles_datadir):
    """ Returns a df of all profiles.index files in path profile_runs_dir.
            Columns are { mass, cb, path_to_LOGS, priority, model_number, profile_number }
        Assumes particular path structure,
            as written by mesa_wimps script ./bash_scripts/profile_run.sh
    """
    pidx_cols = ['model_number', 'priority', 'profile_number']

    dflist = []
    for cb in os.listdir(profile_runs_dir):
        if cb[0] == '.': continue # skip hidden directories
        cbpath = profile_runs_dir+ cb
        for mass in os.listdir(cbpath):
            if mass[0] == '.': continue # skip hidden directories
            Lpath = cbpath+'/'+ mass+'/'+'LOGS'
            fpidx = Lpath+ '/profiles.index'
            try:
                df = pd.read_csv(fpidx, names=pidx_cols, skiprows=1, header=None, sep='\s+')
            except:
                strsplit = fpidx.split('/') # don't print the whole path
                print('Skipping {cb} dir {s}'.format(cb=cb, s=strsplit[-3]))
                continue

#             add mass, cb, and path
            tmp = mass.split('_', maxsplit=1)[0][1:]
            mass_float = float(tmp[0]+'.'+tmp[2:])
            df['mass'] = mass_float
            df['cb'] = int(cb[1])
            df['path_to_LOGS'] = Lpath

            dflist.append(df)

    return pd.concat(dflist, ignore_index=True)
try:
    pidxdfOG
    print('pidxdfOG dataframe already exists.')
except:
    print('Loading pidxdfOG: df of all profiles.index files in path {}'.format(profiles_datadir))
    pidxdfOG = get_profilesindex_df()

def get_meta_df(pidxdf=None):
    """ Returns df of metadata of all profiles in pidxdf
        Looks up path to files in pidxdf (as returned by get_profilesindex_df().
    """
#     Get correct rows from pidxdf
    try:
        p = pidxdf if pidxdf is not None else pidxdfOG
        assert type(p)==pd.DataFrame

    except:
        print('Getting get_profilesindex_df()...')
        p = get_profilesindex_df()

    dflist = []
    # Get the metadata:
    for i, row in p.iterrows():
        fprof = get_profile_path(row)
        try:
            metadf = pd.read_csv(fprof, header=1, sep='\s+', nrows=1)
        except:
            print(fprof, 'metadata cannot be loaded')

        idx = get_pidx(row.cb, row.model_number)
        metadf['pidx'] = str(idx)
        metadf['cb'] = row.cb
        metadf['path'] = fprof

        dflist.append(metadf)
#         break

    df = pd.concat(dflist, ignore_index=True)
    df.set_index('pidx', inplace=True, drop=False)
    return df
try:
    mdf
    print('mdf dataframe already exists.')
except:
    print('Loading mdf: df of metadata of all profiles in pidxdf')
    mdf = get_meta_df()

def load_prof_from_file(cb, modnum, mass=3.5, pidxdf=None):
    """ Returns profile index and 2 dfs (df and metadf) from profile#.data info.
        Looks up path to file in pidxdf (as returned by get_profilesindex_df().
    """
#     Get correct rows from pidxdf
    try:
        p = pidxdf if pidxdf is not None else pidxdfOG
        assert type(p)==pd.DataFrame

    except:
        print('Getting get_profilesindex_df()...')
        p = get_profilesindex_df()
    req = p.loc[(p.mass==mass) & (p.cb==cb) & (p.model_number==modnum)] # df with rows matching input
    if len(req)>1:
        print('Found {n} rows matching input m{mass} c{cb} mod{mod}. Paths are: \
                \n{path}\n\nUsing the last.'
              .format(n=len(req), mass=mass, cb=cb, mod=modnum, path=req.path_to_LOGS.values))

#     Load data and meta_data to dfs
    try:
        row = req.iloc[-1]
        fprof = get_profile_path(row) # Get path to profile#.data

        df = pd.read_csv(fprof, header=4, sep='\s+')
        df['path'] = fprof
        metadf = pd.read_csv(fprof, header=1, sep='\s+', nrows=1)
        metadf['path'] = fprof
    except:
        print('Profile for m{mass}, c{cb}, mod{mod} does not seem to exist.'.format(
                                                        mass=mass, cb=cb, mod=modnum))

    idx = get_pidx(cb, modnum, mass=mass)

    return [idx, df, metadf]

# dictionary of loaded profile#.data dfs
pdfs = {}
# main key: profile index as returned by load_prof_from_file
# main value: dict with keys/values: idx, df, metadf
def get_pdf(cb, modnum, mass=3.5, rtrn='df'):
    """ Returns profile#.data as df. Loads from file if needed.
        rtrn =  'metadf' returns the metadf
                'idx' returns the profile index
                'all' returns all three as a list
    """
    pidx = get_pidx(cb, modnum, mass=mass)
    try:
        idx = pdfs[pidx]['idx']
        df = pdfs[pidx]['df']
        metadf = pdfs[pidx]['metadf']
    except:
        idx, df, metadf = load_prof_from_file(cb, modnum, mass=mass) # load from file
        pdfs[idx] = {} # save to dict
        pdfs[idx]['df'] = df
        pdfs[idx]['metadf'] = metadf

    if rtrn == 'df':
        return df
    if rtrn == 'metadf':
        return metadf
    if rtrn == 'idx':
        return idx
    if rtrn == 'all':
        return [idx, df, metadf]
# fe Profiles setup

# fs History setup
def get_hidx(cb,mass=1.0):
    return 'm{mass}c{cb}'.format(mass=mass, cb=cb)

def load_hist_from_file(cb, mass=1.0, from_file=True, pidxdf=None):
    """ Returns df of history.data from LOGS dir of
            highest model number matching cb, mass (to get most copmlete data).

        from_file = string forces reload from disk using LOGS dir inside dir string.
                    Single dir with no '/' (e.g. 'm1p00_stopmod10000')
                                            gets full path from pidxdf.
                    Any string including a '/' uses this as full path.

        Send pidxdf = [] (or other type) to force it to reload from disk
    """
    hidx = get_hidx(cb, mass=mass)
    print('Loading {} history df from file. Looking for path...'.format(hidx))

    # Get pidxdf
    try:
        p = pidxdf if pidxdf is not None else pidxdfOG
        assert type(p)==pd.DataFrame
    except:
        print('Getting get_profilesindex_df()...')
        p = get_profilesindex_df()

    # Get path to LOGS dir
    try:
        mcb_mods = p[(p.mass==mass) & (p.cb==cb)][['model_number','path_to_LOGS']]

        if type(from_file) == str: # get the one from the dir matching from_file
            # test if full path is given. else get full path from pidxdf
            if len(from_file.split('/')) > 1:
                Lpath = from_file
            else:
                dirlist = list((mcb_mods.groupby('path_to_LOGS')).groups.keys())
                for dl in dirlist:
                    finaldir = dl.split('/')[-2]
                    if finaldir == from_file:
                        Lpath = dl
                        print('Found requested dir: ', from_file)
                        continue

        else: # get the one with the highest model number
            Lpath = mcb_mods.sort_values('model_number').iloc[-1].path_to_LOGS

        assert type(Lpath) == str

    except:
        print('No models matching m{mass} c{cb} dir {ff} in pidxdf'.format(mass=mass, cb=cb, ff=from_file))
        print('Specific dir requested: {ff}'.format(ff=from_file))

    # Load history.data
    hpath = Lpath+ '/history.data'
    print('Loading history df from path {}'.format(hpath))
    try:
        df = pd.read_csv(hpath, header=4, sep='\s+')
    except:
        print(hpath, 'not loaded')

    print()

    return df

hdfs = {} # holds loaded history dataframes
def get_hdf(cb, mass=1.0, from_file=False):
    """ Returns history.data file as df from loaded dict.
        Loads data from file if necessary, getting the file
            with the largest model number in pidxdf.

        from_file == True forces reload from disk using LOGS dir with
                highest model number matching cb, mass (to get most complete data).
        from_file = string forces reload from disk using LOGS dir inside dir string.
                    Single dir with no '/' (e.g. 'm1p00_stopmod10000')
                                            gets full path from pidxdf.
                    Any string including a '/' uses this as full path.
    """

    hidx = get_hidx(cb, mass=mass)
    try:
        hdf = hdfs[hidx]
        if from_file != False:
            hdf = [] # change type to force reload from file
        assert type(hdf) == pd.DataFrame
    except:
        hdf = load_hist_from_file(cb, mass=mass, from_file=from_file, pidxdf=[])
        hdf['cb'] = cb
        hdf['hidx'] = hidx
        hdf.set_index('hidx', drop=False, inplace=True)

        # do not write to master hdfs if from_file is a string with single dir
        # (used for burning cols)
        if type(from_file) == str:
            split = from_file.split('/')
            # len(split)>1 if from_file = full path to dir
            # (used to get full history file from r2tf_dir. Want to store this.)
        if (type(from_file) != str) or (len(split)>1):
            hdfs[hidx] = hdf

    return hdf


def get_r2tf_LOGS_dirs(masses=[1.0, 3.5], cbs=[0,3,6]):
    """ Used to load hdfs dict in plot_* functions.
            e.g. plot argument from_file = pf.get_r2tf_LOGS_dirs(masses=mlist, cbs=cblist)
        Returns dict with
            key = get_hidx(cb,mass) (this returns: m(mass)c(cb)),
            value = full path: r2tf_dir/(cb)/(mass)/LOGS
    """

    path_dict = {}
    for m in masses:
        for c in cbs:
            key = get_hidx(c,m)
            mprec = str(format(m, '.2f')) # ensure we keep 2 decimal places
            mstr = mprec[0] + 'p' + mprec[-2:]
            path = r2tf_dir+ '/c{}/m{}/LOGS'.format(c, mstr)

            try: # make sure the path exists
                assert os.path.isdir(path)
                path_dict[key] = path
            except:
                print('\nWARNING, path does not exist: {}\n'.format(path))

    return path_dict


# fe History setup

# fs Isochrone setup
def load_isos_from_file(fin=iso_csv, cols=None):
    if cols is None:
        cols = ['PrimaryEEP', 'EEP', 'log10_isochrone_age_yr', 'initial_mass',
                'log_Teff', 'log_L', 'log_center_T', 'log_center_Rho', 'cboost']

    try:
        isodf = pd.read_csv(fin, header=0, sep=',', usecols=cols)
    except:
        print()
        print('Problem reading file', fin)
        print('isochrones.csv could not be loaded to df.')
        print()


    isodf = isodf.astype({'PrimaryEEP':int, 'EEP':int, 'cboost':int})

    return isodf

def get_iso_ages(isodf):
    """ Returns list of iso ages that can be plotted"""
    grp = isodf.groupby('log10_isochrone_age_yr')
    return list(grp.groups.keys())

def get_isoage_label(isoage):
    """ Converts log to a*10^b"""
    age = r'{age:5.5e}'.format(age = 10**isoage) # get scientific notation

    if int(age[-1]) == 0:
        tmp = r'${b}\times '.format(b=age[:3]) # separate steps so latex will parse correctly
        agestr = tmp + '10^{10}$'
    else:
        agestr = r'${b}\times 10^{p}$'.format(b=age[:3], p=age[-1])

    return r'{age} [yrs]'.format(age=agestr)
# fe Isochrone setup

# fs delta Tau plot
def get_descdf(fin=fdesc):

    descdf = pd.read_csv(fin)

    print('fixing mass precision in {}'.format(fin))
    descdf = fix_desc_mass(descdf) # fix problem with mass precision
    descdf.set_index('star_index', inplace=True)

    return descdf

def plot_delta_tau(descdf, save=None):
    if save is None:
        plt.figure()
        mnum=5e2
    else:
        plt.figure(figsize=(savefigw, savefigh))
        mnum=1e4
    ax = plt.gca()
    cbgroup = descdf.sort_values('mass').groupby(('cboost'))
    for i, (cb,dat) in enumerate(cbgroup):
        if cb == 0:
            continue
        mass, mst = interp_mstau(dat.mass, dat.MStau, mnum=mnum)
        cmapdict = get_cmapdict(cb,len(mass))
        plt.scatter(mass,mst(mass), s=5, **cmapdict)
#         plt.scatter(dat.mass,dat.MStau, c=cb*np.ones(len(dat.mass)), cmap=cbcmap, vmin=0, vmax=6, s=10)

#         color = get_cmap_color(cb)
#         plt.plot(mass,mst(mass), color=color)
#         plt.plot(dat.mass,dat.MStau, color=color)

    plt.axhline(0., color=get_cmap_color(0), lw=1)
    cbar = get_cbcbar()

#     Axes
    plt.semilogx()
    ax.xaxis.set_major_locator(plt.MultipleLocator(1.0))
    # ax.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    plt.minorticks_on()
    ax.xaxis.set_minor_locator(plt.MultipleLocator(0.1))
    # ax.xaxis.set_minor_formatter(FormatStrFormatter(""))


    plt.xlabel(r'Star Mass [M$_{\odot}$]')
    plt.ylabel(r'$\Delta \tau_{\mathrm{MS}}\ /\ \tau_{\mathrm{MS,NoDM}}$')
#     plt.ylabel(r'$\frac{\Delta \tau_{MS}}{\tau_{MS,\ Fid}}$')
#     plt.title('Change in Main Sequence Lifetimes')
    plt.tight_layout()
    if save is not None: plt.savefig(save)
    plt.show()



def interp_mstau(mass, MStau, mnum=1e3):
    mst = interp1d(mass,MStau)
    mass = np.linspace(0.8,5.0,int(mnum))
    return [mass, mst]

# fe delta Tau plot

# fs Teff plot
def plot_Teff(mlist=None, cblist=None, from_file=False, save=None):
    """ To load history.data from original MESA output dir r2tf_dir,
                                use from_file = get_r2tf_LOGS_dirs()
    """

    cut_axes=True
    if mlist is None:
        mlist = [1.0, 3.5]
    if cblist is None:
        cblist = [0,6]

    mcolor = [get_cmap_color(cmap_masses[m], cmap=mcmap, myvmin=mvmin, myvmax=mvmax) \
               for m in mlist ] # plot colors

    nrows = 1
    ncols = len(cblist)

    if save is None:
        f, axs = plt.subplots(sharex=True, sharey=True, nrows=nrows,ncols=ncols)
    else:
        f, axs = plt.subplots(sharex=True, sharey=True, nrows=nrows,ncols=ncols,
                              figsize=(savefigw+7, savefigh))

    for a, cb in enumerate(cblist):
        for im, mass in enumerate(mlist):
            fromf = from_file if type(from_file)!=dict else from_file[get_hidx(cb,mass)]
            hdf = get_hdf(cb,mass=mass, from_file=fromf)

            fromf0 = from_file if type(from_file)!=dict else from_file[get_hidx(0,mass)]
            hdf0 = get_hdf(0,mass=mass, from_file=fromf0)

            if cut_axes:
                hdf = cut_HR_hdf(hdf) # cut preMS
                hdf0 = cut_HR_hdf(hdf0)

            age = hdf.star_age - hdf.star_age.iloc[0]
            age.iloc[0] = 1
            axs[a].plot(np.log10(age), hdf.log_Teff, \
                                    c=mcolor[im], zorder=-3*mass+16)
            if a != 0:
                age = hdf0.star_age - hdf0.star_age.iloc[0]
                age.iloc[0] = 1
                axs[a].plot(np.log10(age), hdf0.log_Teff, zorder=-3*mass+17, c='w', lw=1.5)
                axs[a].plot(np.log10(age), hdf0.log_Teff, zorder=-3*mass+18, c=mcolor[im], lw=1)

        # Title panels
        if cb==0:
            lbl = r'NoDM'
        else:
            lbl = r'$\Gamma_B = 10^{}$'.format(cb)
        axs[a].annotate(lbl,(ann_rmarg,ann_tmarg), fontsize=ann_fs, xycoords='axes fraction')

        # Axes labels:
        axs[a].set_xlabel(r'log (Star Age/yr)')
    teff = r'log ($T_{eff}$ /K)'
    axs[0].set_ylabel(teff)


    # Colorbar
    cb_top = 0.99; cb_bot = 0.12; cb_right = 0.92; cb_left = 0.06
    cb_wspace = 0.0; cb_hspace = 0.2
    add_axes_list = [cb_right+0.005, cb_bot, 0.02, cb_top-cb_bot]
                            # (pushes right, pushes up, width, height)
    cmapdict = get_cmapdict(cmap_masses[mass],len([1]), which_cmap='mcmap')
    ascat = axs[a].scatter(6,3.5, marker='+', s=0.01, **cmapdict)
    f.subplots_adjust(bottom=cb_bot, top=cb_top, left=cb_left, right=cb_right,
                        wspace=cb_wspace, hspace=cb_hspace)
    cb_ax = f.add_axes(add_axes_list)
    cbar = get_mcbar(sm=ascat, cax=cb_ax, f=f)

#     plt.tight_layout() # makes colorbar overlap plot
    if save is not None: plt.savefig(save)
    plt.show()
# fe Teff plot

# fs HR tracks plot
def plot_HR_tracks(mlist=None, cblist=None, from_file=False, cut_axes=True, save=None):
    """ To load history.data from original MESA output dir r2tf_dir,
                                use from_file = get_r2tf_LOGS_dirs()
    """

    if mlist is None:
        mlist = [1.0, 3.5]
    if cblist is None:
        cblist = [0,6]

    mcolor = [get_cmap_color(cmap_masses[m], cmap=mcmap, myvmin=mvmin, myvmax=mvmax) \
               for m in mlist ] # plot colors

    nrows = 1
    ncols = len(cblist)

    if save is None:
        f, axs = plt.subplots(sharex=True, sharey=True, nrows=nrows,ncols=ncols)
    else:
        f, axs = plt.subplots(sharex=True, sharey=True, nrows=nrows,ncols=ncols,
                              figsize=(savefigw+7, savefigh))

    for a, cb in enumerate(cblist):
        for im, mass in enumerate(mlist):
            fromf = from_file if type(from_file)!=dict else from_file[get_hidx(cb,mass)]
            hdf = get_hdf(cb,mass=mass, from_file=fromf)

            fromf0 = from_file if type(from_file)!=dict else from_file[get_hidx(0,mass)]
            hdf0 = get_hdf(0,mass=mass, from_file=fromf0)

            if cut_axes:
                hdf = cut_HR_hdf(hdf) # cut preMS
                hdf0 = cut_HR_hdf(hdf0)

            axs[a].plot(hdf.log_Teff, hdf.log_L, c=mcolor[im])
            if a != 0:
                axs[a].plot(hdf0.log_Teff, hdf0.log_L, zorder=9, c='w', lw=1.5)
                axs[a].plot(hdf0.log_Teff, hdf0.log_L, zorder=10, c=mcolor[im], lw=1)

        # Title panels
        if cb==0:
            lbl = r'NoDM'
        else:
            lbl = r'$\Gamma_B = 10^{}$'.format(cb)
        axs[a].annotate(lbl,(ann_lmarg,ann_tmarg), fontsize=ann_fs, xycoords='axes fraction')
        # Axes invert:
        axs[a].invert_xaxis()
        # Axes labels:
        if a == 0:
            axs[a].set_ylabel(r'log (L / L$_{\odot}$)')
        teff = r'log (T$_{\mathrm{eff}}$ / K)'
        axs[a].set_xlabel(teff)


    # Colorbar
    cb_top = 0.99; cb_bot = 0.12; cb_right = 0.92; cb_left = 0.06
    cb_wspace = 0.0; cb_hspace = 0.2
    add_axes_list = [cb_right+0.005, cb_bot, 0.02, cb_top-cb_bot]
                            # (pushes right, pushes up, width, height)
    cmapdict = get_cmapdict(cmap_masses[mass],len([1]), which_cmap='mcmap')
    ascat = axs[a].scatter(4,2.3, marker='+', s=0.01, **cmapdict)
    f.subplots_adjust(bottom=cb_bot, top=cb_top, left=cb_left, right=cb_right,
                        wspace=cb_wspace, hspace=cb_hspace)
    cb_ax = f.add_axes(add_axes_list)
    cbar = get_mcbar(sm=ascat, cax=cb_ax, f=f)


    if save is not None: plt.savefig(save)
    plt.show()
# fe HR tracks plot

# fs HR isochrones plot
def get_midx_dict(indf, mtarg=None, mtol=None):
    """ Returns dict of key = mtarg, value = list of df index where initial_mass is closest to mtarg, one for each age.
        Returns None if found mass is not withint mtol of mtarg.

    """

    if mtarg is None:
        mtarg = [1.0, 3.5]

    if mtol is None:
        mtol = 0.1

    agedf = indf.groupby('log10_isochrone_age_yr')

    midx_dict = {}
    for mt in mtarg:
        midx_list = []
        for i, (age, df) in enumerate(agedf):
            # find mass closest to mtarg:
            midx = df.iloc[(df['initial_mass']-mt).abs().argsort()[:1]].index
            midx = list(midx)[0] # get the value of the index

            mdiff = np.abs(df.loc[midx].initial_mass - mt)
            if mdiff > mtol:
                midx = None # return none if found mass is not close enough to target
            midx_list.append(midx)

        midx_dict[mt] = midx_list

    return midx_dict



def plot_isos(isodf, plot_times=None, cblist=None, cut_axes=False, save=None):
    """ Plots isochrone panels.
    """
    if cblist is None:
        cblist = [0,3,6]

    nrows = 1
    ncols = len(cblist)

    if save is None:
        f, axs = plt.subplots(sharex=True, sharey=True, nrows=nrows,ncols=ncols)
    else:
        f, axs = plt.subplots(sharex=True, sharey=True, nrows=nrows,ncols=ncols,
                              figsize=(savefigw+7, savefigh))
    axs = axs.flatten() # flatten the array for later iteration

    # Separate isodf by cboost
    cbdfg = isodf.groupby('cboost')
    cbdf0 = cbdfg.get_group(0)
    for a, cb in enumerate(cblist):
        cbdf = cbdfg.get_group(cb)
        if plot_times is not None:
            cbdf = cbdf[cbdf.log10_isochrone_age_yr.isin(plot_times)]
            cbdf0 = cbdf0[cbdf0.log10_isochrone_age_yr.isin(plot_times)]

        p = axs[a].scatter(cbdf.log_Teff, cbdf.log_L, zorder=1,
               c=cbdf.log10_isochrone_age_yr, cmap=isocmap, vmin=isovmin, vmax=isovmax)

        if a != 0:
            axs[a].scatter(cbdf0.log_Teff, cbdf0.log_L, zorder=2, s=1.25, c='w')
#                    c=cbdf0.log10_isochrone_age_yr, cmap=isocmap, vmin=isovmin, vmax=isovmax)
            axs[a].scatter(cbdf0.log_Teff, cbdf0.log_L, zorder=3, s=0.75,
                   c=cbdf0.log10_isochrone_age_yr, cmap=isocmap, vmin=isovmin, vmax=isovmax)

        # Highlight some masses:
#             if cb not in [0,6]: continue # skip this part
        midx_dict = get_midx_dict(cbdf, mtol=1e-1)
        for mkey, midx in midx_dict.items():
            if midx is None: continue
            mrkr = '^' if mkey == 3.5 else 'o'
#             kwargs = {'marker':mrkr, 'facecolors':'none', 'edgecolors':'k', \
#                       's':100, 'zorder':20}
            ylw = normalize_RGB((255,255,0))
            kwargs = {'marker':mrkr, 'facecolors':'none', 'edgecolors':ylw, \
                      's':100, 'zorder':20}
            rows = cbdf.loc[midx] # get the correct rows from the df
#                     axs[a].scatter(mdf.log_Teff, mdf.log_L, color=cbc[cb], marker=mrkr, s=100)
            axs[a].scatter(rows.log_Teff, rows.log_L,
                           c=rows.log10_isochrone_age_yr, cmap=isocmap, vmin=isovmin, vmax=isovmax,
                           **kwargs)

        # Title panels
        if cb==0:
            lbl = r'NoDM'
        else:
            lbl = r'$\Gamma_B = 10^{}$'.format(cb)
        axs[a].annotate(lbl,(ann_lmarg,ann_tmarg), fontsize=ann_fs, xycoords='axes fraction')
        # Axes params
        axs[a].invert_xaxis()
#         axs[a].grid(linestyle='-', linewidth='0.5', color='0.7')

    # Axes labels
        teff = r'log (T$_{\mathrm{eff}}$ / K)'
        axs[a].set_xlabel(teff)
    axs[0].set_ylabel(r'log (L / L$_{\odot}$)')

    # Axes limits
    axs[0].set_xlim(4.15,3.45)
    axs[0].set_ylim(-0.5,4)

    # Colorbar
    cb_top = 0.98; cb_bot = 0.12; cb_right = 0.91; cb_left = 0.07
    cb_wspace = 0.0; cb_hspace = 0.2
    add_axes_list = [cb_right+0.005, cb_bot, 0.02, cb_top-cb_bot]
                            # (pushes right, pushes up, width, height)
    f.subplots_adjust(bottom=cb_bot, top=cb_top, left=cb_left, right=cb_right,
                        wspace=cb_wspace, hspace=cb_hspace)
    cb_ax = f.add_axes(add_axes_list)
    cbar = get_isocbar(sm=p, cax=cb_ax, f=f)

#     plt.tight_layout()
    if save is not None: plt.savefig(save)
    plt.show()

# fe HR isochrones plot

# fs Kipp plot
def get_burn_cols(hdf):
    """ Returns df with cols {star_age, burn_type, mass_top, mass_bottom}
                where mass_ is mass coord in [Msun] (converts q to mass(<r)).
        Ensures burn zone matches burn type.
        """
#     concat burn columns, check that have 1 time step for every burn type

    hdf.set_index('star_age', inplace=True, drop=False)

    btcols = sorted([ col for col in hdf.columns if 'burn_type' in col ])
    bqtcols = sorted([ col for col in hdf.columns if 'burn_qtop' in col ])
    dflist = []
    for i in range(len(btcols)):
        bt_name = btcols[i]
        bqt_name = bqtcols[i]

        bt = hdf[bt_name]
        ages = hdf.star_age

        # get burn zone coords
        burn_qtop = hdf[bqt_name]
        if i == 0:
            burn_qbottom = 0*burn_qtop
        else:
            burn_qbottom = hdf[bqtcols[i-1]]
        # convert to Msun
        mass_top = burn_qtop*hdf.star_mass
        mass_bottom = burn_qbottom*hdf.star_mass

        col_list = OD([('star_age',ages), ('burn_type',bt),
                       ('mass_top',mass_top), ('mass_bottom',mass_bottom)])
#         print([col_list[i].sample() for i in range(len(cols))])
        tmp = pd.DataFrame(col_list)
#         tmp.columns = cols


        dflist.append(tmp)


    df = pd.concat(dflist, axis=0)
    df = df[df.burn_type != -9999]
#     print(btcols)
#     'burn_type_1', 'burn_qtop_1'
    return df

# ### NOT CURRENTLY USING
# def interp_burn_cols(df, xlim):
#     """ Returns dataframe interpolated from df
#     """
#     sa = df.star_age.values
#     ages = np.linspace(xlim[0], xlim[1], int(1e6))
#     fb = interp1d(sa,df.mass_bottom)
#     ft = interp1d(sa,df.mass_top)
#
#     icols = OD([ ('star_age',ages),
#                 ('mass_bottom', fb(ages)), ('mass_top',ft(ages)) ])
#     idf = pd.DataFrame(icols)
#
#     return idf

# fe Kipp plot
