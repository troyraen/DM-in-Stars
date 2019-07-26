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
import matplotlib.gridspec as gs
import matplotlib.colors as colors
from matplotlib.colors import ListedColormap
from matplotlib.ticker import FormatStrFormatter
import os
# fe imports

# fs Set plot defaults:
# mpl.rcParams.update(mpl.rcParamsDefault)
print('Setting plot defaults.')

mpl.rcParams['figure.figsize'] = [8.0, 5.0]
savefigw = 13
savefigh = 6
savefigw_vert = 8
savefigh_vert = 3 # multiply this by number of times plotted (outer rows)

mpl.rcParams['font.size'] = 13
mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['font.family'] = 'DejaVu Sans' #'cmu serif'

mpl.rcParams['lines.linewidth'] = 2
# mpl.rcParams['legend.fontsize'] = 'small'
plt.rcParams["text.usetex"] = False
# mpl.rcParams['axes.linewidth'] = 0.8
mpl.rcParams['axes.titlesize'] = 20
mpl.rcParams['axes.titleweight'] = 'normal'
mpl.rcParams['axes.labelsize'] = 20
# mpl.rcParams['axes.labelweight'] = 'heavy'
mpl.rcParams['xtick.labelsize'] = 15
mpl.rcParams['xtick.direction'] = 'inout'
mpl.rcParams['xtick.major.size'] = 8
mpl.rcParams['xtick.minor.visible'] = True
mpl.rcParams['xtick.minor.size'] = 4.0
mpl.rcParams['ytick.labelsize'] = 15
mpl.rcParams['ytick.direction'] = 'inout'
mpl.rcParams['ytick.major.size'] = 8
mpl.rcParams['ytick.minor.visible'] = True
mpl.rcParams['ytick.minor.size'] = 4.0

### Annotations
# This is probably the same for all plots
ann_fs = 26 # font size
ann_rmarg = 0.97 # use for right align
ann_lmarg = 0.03 # use for left align
ann_tmarg = 0.97 # top margin

# Used to adjust axes
cb_top = 0.99; cb_bot = 0.12; cb_right = 0.92; cb_left = 0.06
cb_wspace = 0.0; cb_hspace = 0.2

def adjust_plot_readability(fig=None, fontAdj=False, fontOG=None, plot=None):
    """ Adjusts font sizes and axes positions for multi-panel plots.
        This function must be called three times, once for each option:

        fontAdj (bool): call before creating the figure with
                        fontOG = adjust_plot_readability(fontAdj=True)
        fig (matplotlib figure object): call after creating figure and colorbar to adjust axes.
                        f = adjust_plot_readability(fig=f)
        fontOG (dict): original mpl.rcParams to be reset after plotting.
                        adjust_plot_readability(fontOG=fontOG)

        plot (string): one of 'teff', 'tracks', 'isos' determines font sizes and axes spacing.

        Returns:
            if fontAdj is True, returns a dict with original mpl.rcParams.
                Pass this dict as the fontOG param to reset defaults after plotting.
            if fig is not None, returns fig (after adjusting)
    """

    if fontOG is not None:
        # Reset font sizes
        for key, val in fontOG.items():
            mpl.rcParams[key] = val
        return None

    if fontAdj:
        # Adjust font sizes
        if plot in ['teff', 'tracks']:
            rcp = {
                    # 'axes.titlesize': ann_fs,
                    # 'axes.labelsize': ann_fs+5,
                    # 'xtick.labelsize': ann_fs-5,
                    # 'ytick.labelsize': ann_fs-5
                    }
        elif plot == 'isos':
            rcp = {
                    # 'axes.titlesize': ann_fs,
                    'axes.labelsize': ann_fs+5,
                    'xtick.labelsize': ann_fs-5,
                    'ytick.labelsize': ann_fs-5
                    }
        elif plot == 'm3.5':
            rcp = {
                    # 'axes.titlesize': ann_fs,
                    'axes.labelsize': ann_fs+5,
                    'xtick.labelsize': ann_fs-5,
                    'ytick.labelsize': ann_fs-5
                    }
        else:
            assert False, "adjust_plot_readability() received invalid option plot = {}".format(plot)

        rcpOG = {}
        for key, val in rcp.items():
            rcpOG[key] = mpl.rcParams[key]
            mpl.rcParams[key] = rcp[key]

        return rcpOG

    if fig is not None:
        # Adjust axes
        if plot=='isos':
            eps = 0.05
            epsr = 0.07
            epsb = -eps
            epst = 0
            wspace = cb_wspace
            hspace = cb_hspace
        if plot=='tracks':
            eps = 0.08
            epsr = eps
            epsb = 0
            epst = 0
            wspace = cb_wspace
            hspace = 0
        else:
            eps = 0.005
            epsr = eps
            epsb = eps
            epst = 0
            wspace = cb_wspace
            hspace = cb_hspace

        fig.subplots_adjust(bottom=cb_bot+epsb, top=cb_top-epst, left=cb_left+eps, \
                            right=cb_right-epsr, wspace=wspace, hspace=hspace)

        return fig, [eps, epsr, epsb, epst]

    # If we get here, there were no valid arguments passed in
    assert False, "No valid options were passed to adjust_plot_readability()"
    return None


# fe Set plot defaults

# fs Files and directories
mesaruns = '/Users/troyraen/Osiris/mesaruns'
datadir = mesaruns+ '/RUNS_2test_final/plotsdata/Glue'
plotdir = '/Users/troyraen/Documents/Comms/WIMP_Paper'
# finalplotdir = '/Users/troyraen/Osiris/mesaruns/RUNS_2test_final/plots'
# finalplotdir = '/Users/troyraen/Google_Drive/MESA/code/mesa_wimps/final_plots'
finalplotdir = '/Users/troyraen/Google_Drive/MESA/code/mesa_wimps/DMS-Paper/plots'
fzams = datadir+ '/zams.csv'
# fdesc = datadir+ '/descDF_MS_allmassesreg.csv'
fdesc = datadir+ '/descDF_updated.csv'
profiles_datadir = mesaruns+ '/RUNS_2test_final/profile_runs/'
iso_csv = mesaruns+ '/RUNS_2test_final/isochrones/isochrones.csv'
hotTeff_csv = mesaruns+ '/RUNS_2test_final/isochrones/hotTeff.csv'
r2tf_dir = mesaruns+ '/RUNS_2test_final'

try: # mount Osiris dir if not already
    assert os.path.isdir(mesaruns)
    print('Osiris dir is already mounted.')
except:
    mounto = int(input('Do you want to mount Osiris? (1 = yes)') or 0)
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
    h1cuts, __ = get_h1cuts()
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

    plot_labels = OD([('ZAMS', r'ZAMS'),
                 ('IAMS', r'$X_{\rm{c}} = 0.3$'),
                 ('H-1', r'$X_{\rm{c}} = 0.1$'),
                 ('H-2', r'$X_{\rm{c}} = 10^{-2}$'),
                 ('H-3', r'$X_{\rm{c}} = 10^{-3}$'),
                 ('H-4', r'$X_{\rm{c}} = 10^{-4}$'),
                 ('H-6', r'$X_{\rm{c}} = 10^{-6}$'),
                 ('TAMS', r'$X_{\rm{c}} = 10^{-12}$')])

    return h1cuts, plot_labels

def cut_HR_hdf(hdf, cuts=['ZAMS'], tahe=[]):
    """ To cut at TACHeB, must send tahe = [descdf,mass,cb] for HR tracks
    """

    df = hdf.copy()
    h1cuts, __ = get_h1cuts()

    if 'ZAMS' in cuts: # cut pre ZAMS
        ZAMS_cut = h1cuts['ZAMS']
        df = df[df.center_h1 < ZAMS_cut]

    if 'ZAMS_time_step' in cuts: # add time_step [years] column, then cut pre ZAMS
        df['time_step'] = df.star_age.diff()
        ZAMS_cut = h1cuts['ZAMS']
        df = df[df.center_h1 < ZAMS_cut]

    if 'IAMS' in cuts: # cut post IAMS
        IAMS_cut = h1cuts['IAMS']
        df = df[df.center_h1 > IAMS_cut]

    if 'H-3' in cuts: # cut post H-3 (leaveMS)
        H3_cut = h1cuts['H-3']
        df = df[df.center_h1 > H3_cut]

    if 'TAMS' in cuts: # cut post TAMS
        TAMS_cut = h1cuts['TAMS']
        df = df[df.center_h1 > TAMS_cut]

    if 'TACHeB' in cuts: # cut post TACHeB
        descdf,mass,cb = tahe
        TACHeB_model = descdf.loc[((descdf.mass==mass) & (descdf.cboost==cb)),'TACHeB_model']
        # print(mass, cb, len(TACHeB_model))
        # print(descdf.head())
        # print(TACHeB_model.iloc[0])
        if len(TACHeB_model)>1:
            print('WARNING: found multiple stars matching mass {} cb {} for TACHeB_model.'\
                    .format(mass, cb))
        # print('mass {}, cb {}, TACHeB_model = {}'.format(mass, cb, TACHeB_model))
        df = df[df.model_number <= TACHeB_model.iloc[0]]

    return df


# fe General helper fncs

###--- COLORMAPS ---###
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
        Intended to be used with plotting that uses mcmap as the colormap.

        Pass cax = subplot axis if using on figure with subplots
        Example:
                f.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.8, \
                                                    wspace=0.02, hspace=0.02)
                cb_ax = f.add_axes([0.83, 0.1, 0.02, 0.8])
                cbar = get_mcbar(sm=ascat, cax=cb_ax, f=f)

    """
    cbar = plt.colorbar(sm, **{'pad':0.01}, **kwargs) if cax is None \
            else f.colorbar(sm, cax=cax, **kwargs, shrink=0.6)
    cbar.set_ticks([i for i in range(len(cmap_masses.keys()))],update_ticks=False)
    cbar.set_ticklabels([m for m in cmap_masses.keys()])
    cbar.ax.minorticks_off()
    cbar.set_label(r'Stellar Mass [M$_{\odot}$]', labelpad=9)
    return cbar

# fe mass colormap

# fs isochrone colormap
# Create iso colormap (isocmap):
# cmap_BuRd = plt.get_cmap('RdBu_r')
# isocmap = truncate_colormap(cmap_BuRd, minval=[0.0,0.52], maxval=[0.48, 1.0], n=500)
cmap_BuPu = plt.get_cmap('BuPu')
isocmap = truncate_colormap(cmap_BuPu, minval=0.25, maxval=1.0, n=500)

# used in plt.plot and other places to normalize colorbar:
isovmin = 7.89
isovmax = 10.15

def get_isocbar(sm=None, cax=None, f=None, ticks=None, **kwargs):
    """ Returns a colorbar that will be added to the side of a plot.
        Intended to be used with plotting that uses isocmap as the colormap.

        cax = subplot axis (if using on figure with subplots)

        ticks = plot_times to set colorbar ticks and labels

        Example:
                f.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.8, \
                                                    wspace=0.02, hspace=0.02)
                cb_ax = f.add_axes([0.83, 0.1, 0.02, 0.8])
                cbar = get_mcbar(sm=ascat, cax=cb_ax, f=f)

    """
    cbar = plt.colorbar(sm, **{'pad':0.01}, **kwargs) if cax is None \
            else f.colorbar(sm, cax=cax, **kwargs)

    if ticks is not None:
        cbar.set_ticks([i for i in ticks],update_ticks=False)
        cbar.set_ticklabels([np.round(i,2) for i in ticks])
    cbar.ax.minorticks_off()
    cbar.set_label(r'log ($Isochrone\ Age$ /yr)', labelpad=5)
    return cbar
# fe isochrone colormap

###--- DATA SETUP ---###
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
        if cb not in ['c0','c1','c2','c3','c4','c5','c6']: continue # skip hidden directories
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
    load = int(input('Do you want to load profiles.index files df? (1 = yes) ') or 0)
    if load == 1:
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
    load = int(input('Do you want to load metadata df? (1 = yes) ') or 0)
    if load == 1:
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

def get_xfrac_lbl(pdf):
    # get h1_center from profile df to use as plot label to check closeness to PEEP definition
    xfrac = pdf[pdf.q == min(pdf.q)].x.iloc[0]
    if xfrac>1e-3:
        xfrac_lbl = r'$h1_c$ = ' + str(np.round(xfrac,4))
    else:
        xfrac_lbl = r'log($h1_c$) = ' + str(np.round(np.log10(xfrac),3))

    return xfrac_lbl

def get_osc_modnums():
    """ Returns OD of c6 model numbers to find profile model numbers.
        Times choosen by eye
        peeps given by:
            get_h1_modnums(frm='history_data', mass=1.0, hdf=get_hdf(6, mass=1.0), cb=6)
    """
    osc_modnums = OD([('Time1',1679),
                 ('Time2',1720),
                 ('Time3',1759),
                 ('Time4',1780),
                 ('Time5',1791),
#                  ('Degen',1915),
                 ('Degen',1909),
                 ('IAMS',1957),
                 ('H-1',2001),
                 ('H-2',2055),
                 ('H-3',2086),
                 ('H-4',2102),
                 ('H-6',2253),
                 ('TAMS',2321)])
    return osc_modnums

# dictionary of loaded profile#.data dfs
try:
    pdfs
except:
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
            highest model number matching cb, mass (to get most complete data).

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

try:
    hdfs
except:
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
        cols = ['PrimaryEEP', 'EEP', 'log10_isochrone_age_yr', 'initial_mass', 'cboost', \
                'log_Teff', 'log_L', 'log_center_T', 'log_center_Rho', \
                'center_h1', 'center_he4']

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

###--- PLOTS ---###
# fs delta Tau plot
def get_descdf(fin=fdesc, fix_mass_prec=False):

    descdf = pd.read_csv(fin)

    if fix_mass_prec:
        print('fixing mass precision in {}'.format(fin))
        descdf = fix_desc_mass(descdf) # fix problem with mass precision

    descdf.set_index('star_index', inplace=True)

    return descdf

def find_cno_trans_mass(df):
    """ Intended to be used as the func arg to .apply(func), as in:
            pp2cno_mass = descdf.groupby('cboost').apply(find_cno_trans_mass)
                            where descdf = pf.get_descdf(fin=pf.fdesc)

            df (DataFrame): as given by df = pf.get_descdf(fin=pf.fdesc)

        Returns:
            df (DataFrame): columns are index and mass of lowest mass for which
                            average MS CNO burning > PP burning.
    """

    # print(df.columns)
    cb = df.cboost.unique()
    if len(cb) != 1:
        print('\nYou must pass a df with a unique cboost value to find_cno_trans_mass(df).\n')
        return

    cno_mass = df.loc[df.CNOavg > df.PPavg].mass
    pp_mass = df.loc[df.CNOavg < df.PPavg].mass

    # make sure all CNO stars are more massive than all PP stars
    assert (cno_mass > pp_mass.max()).all(), \
            "Error: There is a PP star more massive than a CNO star in cb{}.".format(cb[0])

    cmass = cno_mass.min()
    cidx = df.index[df.mass==cmass].values[0]

    return pd.Series({'pp2cno_index':cidx, 'pp2cno_mass':cmass})

def find_cc_trans_mass(df, trans_frac='default', which='avg'):
    """ Intended to be used as the func arg to .apply(func), as in:
            cctrans = descdf.groupby('cboost').apply(find_cc_trans_mass)
                            where descdf = pf.get_descdf(fin=pf.fdesc)

            df (DataFrame): e.g. df = pf.get_descdf(fin=pf.fdesc)

        Returns:
            df (DataFrame): columns are index and mass of lowest mass for which
                            masscc_avg/mass > trans_frac.
    """

    if trans_frac == 'default':
        trans_frac = 0.01

    cb = df.cboost.unique()
    if len(cb) != 1:
        print('\nYou must pass a df with a unique cboost value to find_cc_trans_mass(df).\n')
        return

    # ccdf = df.loc[:,['mass','masscc_avg']]
    # ccdf['ccfrac'] = ccdf.masscc_avg/ccdf.mass
    ccmass_frac = df.masscc_avg/df.mass if which == 'avg' else df.masscc_ZAMS/df.mass
    cctrans_mass = df.loc[ccmass_frac > trans_frac, 'mass'].min()
    ccidx = df.index[df.mass==cctrans_mass].values[0]

    return pd.Series({'cctrans_index':ccidx, 'cctrans_mass':cctrans_mass})


def plot_delta_tau(descdf, cctrans_frac='default', which='avg', save=None):
    if save is None:
        plt.figure()
        mnum=5e2
    else:
        plt.figure(figsize=(savefigw, savefigh*1.1))
        mnum=1e4
    ax = plt.gca()
    cbgroup = descdf.sort_values('mass').groupby(('cboost'))
    pp2cno = cbgroup.apply(find_cno_trans_mass) # df with index 'cboost'
    cctrans = cbgroup.apply(find_cc_trans_mass, trans_frac=cctrans_frac, which=which)
    for i, (cb,dat) in enumerate(cbgroup):
        if cb == 0:
            continue

        cmapdict = get_cmapdict(cb,len(dat.mass))
        plt.scatter(dat.mass,dat.MStau, s=5, **cmapdict)
        plt.plot(dat.mass,dat.MStau, c=get_cmap_color(cb))
        # Plot interpolated MStau:
        # mass, mst = interp_mstau(dat.mass, dat.MStau, mnum=mnum)
        # cmapdict = get_cmapdict(cb,len(mass))
        # plt.scatter(mass,mst(mass), s=5, **cmapdict)

        # plot mass of transition to core convection
        ccidx = int(cctrans.loc[cctrans.index == cb, 'cctrans_index'])
        cmapdict = get_cmapdict(cb,len([ccidx]))
        plt.scatter(dat.loc[ccidx,'mass'], dat.loc[ccidx,'MStau'], \
            marker='^', edgecolors='w', s=200, linewidths=0.5, **cmapdict, zorder=100)

    # plot mass of transition to CNO burning
        # cidx = int(pp2cno.loc[pp2cno.index == cb, 'pp2cno_index'])
        # cmapdict = get_cmapdict(cb,len([cidx]))
        # plt.scatter(dat.loc[cidx,'mass'], dat.loc[cidx,'MStau'], \
        #         marker='*', edgecolors='w', s=200, linewidths=0.5, **cmapdict, zorder=100)

    plt.axhline(0., color=get_cmap_color(0), lw=1)
    # cidx = int(pp2cno.loc[pp2cno.index == 0,'pp2cno_index'])
    # plt.scatter(descdf.loc[cidx,'mass'], descdf.loc[cidx,'MStau'], \
            # marker='*', edgecolors='w', s=200, linewidths=0.5, **cmapdict, zorder=99)
    ccidx = int(cctrans.loc[cctrans.index == 0, 'cctrans_index'])
    cmapdict = get_cmapdict(0,len([ccidx]))
    plt.scatter(descdf.loc[ccidx,'mass'], descdf.loc[ccidx,'MStau'], \
                marker='^', edgecolors='w', s=200, linewidths=0.5, **cmapdict, zorder=100)

    cbar = get_cbcbar()

#     Axes
    plt.semilogx()
    ax.xaxis.set_major_locator(plt.MultipleLocator(1.0))
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    plt.minorticks_on()
    ax.xaxis.set_minor_locator(plt.MultipleLocator(0.1))
    ax.xaxis.set_minor_formatter(FormatStrFormatter(""))


    plt.xlabel(r'Stellar Mass [M$_{\odot}$]')
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

def plot_convcore(descdf, which='avg', save=None):
    """ which (string): one of 'avg' or 'ZAMS'
    """
    if save is None:
        plt.figure()
        mnum=5e2
    else:
        plt.figure(figsize=(savefigw, savefigh))
        mnum=1e4
    ax = plt.gca()

    df = descdf.copy()
    df['masscc'] = df.masscc_avg/df.mass if which=='avg' else df.masscc_ZAMS/df.mass
    cbgroup = df.sort_values('mass').groupby(('cboost'))
    # get core convection transition df, index=cboost, cols = [cctrans_index,cctrans_mass]
    cctrans = cbgroup.apply(find_cc_trans_mass, which=which)
    for ckey, cdf in cbgroup:
        c = get_cmap_color(ckey)
        cdf.plot.scatter('mass', 'masscc', ax=ax, label=r'cb{}'.format(str(ckey)), \
                 c=c, logx=True, zorder=np.abs(ckey-6))
        ccidx = int(cctrans.loc[cctrans.index == ckey, 'cctrans_index'])
        plt.scatter(cdf.loc[ccidx,'mass'], cdf.loc[ccidx,'masscc'],
                s=100, c=c, marker='^', zorder=10)

    plt.xlabel('Stellar Mass')
    plt.ylabel(r'$\rm{m}(<r_{\rm{cc}})/\rm{M}_{\star}$')
    ttl = 'CC averaged over MS' if which=='avg' else 'CC at ZAMS'
    plt.title(ttl)
    plt.tight_layout()
    if save is not None: plt.savefig(save)
    plt.show()

    return None



# fe delta Tau plot

# fs Teff plot
def plot_Teff(mlist=None, cblist=None, from_file=False, save=None):
    """ To load history.data from original MESA output dir r2tf_dir,
                                use from_file = get_r2tf_LOGS_dirs()
    """

    fontOG = adjust_plot_readability(fontAdj=True, plot='teff')
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
                              figsize=(savefigw, savefigh))

    for a, cb in enumerate(cblist):
        for im, mass in enumerate(mlist):
            fromf0 = from_file if type(from_file)!=dict else from_file[get_hidx(0,mass)]
            hdf0 = get_hdf(0,mass=mass, from_file=fromf0)

            fromf = from_file if type(from_file)!=dict else from_file[get_hidx(cb,mass)]
            hdf = get_hdf(cb,mass=mass, from_file=fromf)

            if cut_axes:
                hdf = cut_HR_hdf(hdf) # cut preMS
                hdf0 = cut_HR_hdf(hdf0)

            age = hdf.star_age - hdf.star_age.iloc[0]
            age.iloc[0] = 1 # avoid log(0)
            axs[a].plot(np.log10(age), hdf.log_Teff, \
                                    c=mcolor[im], zorder=-3*mass+16)
            if cb != 0:
                age = hdf0.star_age - hdf0.star_age.iloc[0]
                age.iloc[0] = 1 # avoid log(0)
                axs[a].plot(np.log10(age), hdf0.log_Teff, zorder=-3*mass+17, c='w', lw=1.5)
                axs[a].plot(np.log10(age), hdf0.log_Teff, zorder=-3*mass+18, c=mcolor[im], lw=1)

        # Title panels
        if cb==0:
            lbl = r'NoDM'
        else:
            lbl = r'$\Gamma_B = 10^{}$'.format(cb)
        axs[a].annotate(lbl,(ann_rmarg,ann_tmarg), fontsize=ann_fs, xycoords='axes fraction', \
                        horizontalalignment='right', verticalalignment='top')

        # Axes labels and limits:
        axs[a].set_xlabel(r'log ($Stellar Age$/yr)')
    teff = r'log ($T_{\mathrm{eff}}$ /K)'
    axs[0].set_ylabel(teff)
    plt.xlim(6.8, axs[0].get_xlim()[1]-0.4)

    # Colorbar
    f, eps = adjust_plot_readability(fig=f, fontOG=None, plot='teff')
    eps = eps[0]
    add_axes_list = [cb_right-eps+0.005, cb_bot+eps, 0.02, cb_top-cb_bot-eps]
                            # (pushes right, pushes up, width, height)
    cmapdict = get_cmapdict(cmap_masses[mass],len([1]), which_cmap='mcmap')
    ascat = axs[a].scatter(6,3.5, marker='+', s=0.01, **cmapdict)
    cb_ax = f.add_axes(add_axes_list)
    cbar = get_mcbar(sm=ascat, cax=cb_ax, f=f)


#     plt.tight_layout() # makes colorbar overlap plot
    if save is not None: plt.savefig(save)
    plt.show()

    adjust_plot_readability(fig=None, fontOG=fontOG)
    return None

# fe Teff plot

# fs HR tracks plot
def plot_HR_tracks(mlist=None, cblist=None, from_file=False, cut_axes=True, save=None):
    """ To load history.data from original MESA output dir r2tf_dir,
                                use from_file = get_r2tf_LOGS_dirs()
    """

    fontOG = adjust_plot_readability(fontAdj=True, plot='tracks')
    if mlist is None:
        mlist = [1.0, 3.5]
    if cblist is None:
        cblist = [0,6]

    mcolor = [get_cmap_color(cmap_masses[m], cmap=mcmap, myvmin=mvmin, myvmax=mvmax) \
               for m in mlist ] # plot colors

    descdf = get_descdf(fin=fdesc) # get the descdf for ZAMS and TAMS lines

    ncols = 1
    nrows = len(cblist)

    if save is None:
        f, axs = plt.subplots(sharex=True, sharey=True, nrows=nrows,ncols=ncols)
    else:
        f, axs = plt.subplots(sharex=True, sharey=True, nrows=nrows,ncols=ncols,
                                figsize=(savefigw_vert,savefigh_vert*nrows*2))
                              # figsize=(savefigw+7, savefigh))

    for a, cb in enumerate(cblist):
        for im, mass in enumerate(mlist):
            fromf = from_file if type(from_file)!=dict else from_file[get_hidx(cb,mass)]
            hdf = get_hdf(cb,mass=mass, from_file=fromf)

            fromf0 = from_file if type(from_file)!=dict else from_file[get_hidx(0,mass)]
            hdf0 = get_hdf(0,mass=mass, from_file=fromf0)

            if cut_axes:
                hdf = cut_HR_hdf(hdf, cuts=['ZAMS','TACHeB'], tahe=[descdf,mass,cb])
                hdf0 = cut_HR_hdf(hdf0, cuts=['ZAMS','TACHeB'], tahe=[descdf,mass,0])

            axs[a].plot(hdf.log_Teff, hdf.log_L, c=mcolor[im])
            mbool = hdf.model_number== get_h1_modnums(frm='history_data', mass=mass, hdf=hdf, cb=cb)[cb][4]
            # axs[a].scatter(hdf.loc[mbool,'log_Teff'], hdf.loc[mbool,'log_L'], \
            #             c='0.75', s=100, marker='^', zorder=19)
            axs[a].scatter(hdf.loc[mbool,'log_Teff'], hdf.loc[mbool,'log_L'], \
                        c=mcolor[im], edgecolors='0.75', linewidths=1, \
                        s=75, marker='^', zorder=20)
            # axs[a].scatter(hdf.loc[mbool,'log_Teff'], hdf.loc[mbool,'log_L'], \
            #             c='w', s=230)
            # print(mass, cb)
            # print('prev h_c = {}'.format(hdf.loc[hdf.model_number==idx-1,'center_h1']))
            # print('h_c = {}'.format(hdf.loc[hdf.model_number==idx,'center_h1']))

            if cb != 0:
                axs[a].plot(hdf0.log_Teff, hdf0.log_L, zorder=9, c='w', lw=1.5)
                axs[a].plot(hdf0.log_Teff, hdf0.log_L, zorder=10, c=mcolor[im], lw=1)

        # ZAMS and H-3 lines
        ddf0 = descdf.sort_values('mass').groupby('cboost').get_group(0)
        # print(ddf0.sample(5))
        pdict = {'lw':1, 'ls':'-', 'zorder':-1}
        # z, = axs[a].plot(ddf0.ZAMS_Teff, ddf0.ZAMS_L, c='blue', **pdict)
        # l, = axs[a].plot(ddf0.lAMS_Teff, ddf0.lAMS_L, c='fuchsia', **pdict)
        z, = axs[a].plot(ddf0.ZAMS_Teff, ddf0.ZAMS_L, c='0.5', **pdict)
        l, = axs[a].plot(ddf0.lAMS_Teff, ddf0.lAMS_L, c='0.75', **pdict)
        # t, = axs[a].plot(ddf0.TAMS_Teff, ddf0.TAMS_L, c='indigo', **pdict)

        # Title panels
        if cb==0:
            lbl = r'NoDM'
        else:
            lbl = r'$\Gamma_B = 10^{}$'.format(cb)
        axs[a].annotate(lbl,(ann_lmarg,ann_tmarg), fontsize=ann_fs, xycoords='axes fraction', \
                        verticalalignment='top')
        # Axes labels:
        axs[a].set_ylabel(r'log ($L / \mathrm{L}_{\odot}$)')
        teff = r'log ($T_{\mathrm{eff}}$ /K)'
    axs[a].set_xlabel(teff)

    # Axes invert:
    axs[a].invert_xaxis()

    # Colorbar
    f, eps = adjust_plot_readability(fig=f, fontOG=None, plot='tracks')
    eps = eps[0]
    add_axes_list = [cb_right-eps+0.01, cb_bot+eps*5.5, 0.02, cb_top-cb_bot-eps*5.5]
                            # (pushes right, pushes up, width, height)
    cmapdict = get_cmapdict(cmap_masses[mass],len([1]), which_cmap='mcmap')
    ascat = axs[a].scatter(4,2.3, marker='+', s=0.01, **cmapdict)
    cb_ax = f.add_axes(add_axes_list)
    cbar = get_mcbar(sm=ascat, cax=cb_ax, f=f)

    axs[0].legend([z,l], ['ZAMS',r'$H_c < 10^{-3}$'])
    if save is not None: plt.savefig(save)
    plt.show()

    adjust_plot_readability(fig=None, fontOG=fontOG)
    return None
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

def plot_isos_ind(isodf, plot_times=None, cb=None, cut_axes=True, save=None):
    """ Plots isochrone panels.
    """
    fontOG = adjust_plot_readability(fontAdj=True, plot='isos')
    if cb is None:
        cb = 4

    nrows = 1
    ncols = 1

    if save is None:
        f, axs = plt.subplots(sharex=True, sharey=True, nrows=nrows,ncols=ncols)
    else:
        mult_asp = 1.5
        f, axs = plt.subplots(sharex=True, sharey=True, nrows=nrows,ncols=ncols,
                                figsize=(8.5*mult_asp,9.5*mult_asp))
                              # figsize=(savefigw+7, savefigh+14))
    axs = axs.flatten() if ncols!=1 else [axs] # flatten/create the array for later iteration

    # Separate isodf by cboost
    cbdfg = isodf.groupby('cboost')
    cbdf0 = cbdfg.get_group(0)
    for a, cb in enumerate([cb]):
        cbdf = cbdfg.get_group(cb)
        if plot_times is not None:
            cbdf = cbdf[cbdf.log10_isochrone_age_yr.isin(plot_times)]
            cbdf0 = cbdf0[cbdf0.log10_isochrone_age_yr.isin(plot_times)]

            if cut_axes:
                cbdf = cut_HR_hdf(cbdf, cuts=['ZAMS','TACHeB'], tahe=['iso'])
                cbdf0 = cut_HR_hdf(cbdf0, cuts=['ZAMS','TACHeB'], tahe=['iso'])


        p = axs[a].scatter(cbdf.log_Teff, cbdf.log_L, zorder=1,
               c=cbdf.log10_isochrone_age_yr, cmap=isocmap, vmin=isovmin, vmax=isovmax)

        if cb != 0:
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
                      's':300, 'linewidth':4, 'zorder':20}
            rows = cbdf.loc[midx,:] # get the correct rows from the df
#                     axs[a].scatter(mdf.log_Teff, mdf.log_L, color=cbc[cb], marker=mrkr, s=100)
            axs[a].scatter(rows.log_Teff, rows.log_L,
                           c=rows.log10_isochrone_age_yr, cmap=isocmap, vmin=isovmin, vmax=isovmax,
                           **kwargs)

        # Title panels
        if cb==0:
            lbl = r'NoDM'
        else:
            lbl = r'$\Gamma_B = 10^{}$'.format(cb)
        axs[a].annotate(lbl,(ann_lmarg,ann_tmarg), fontsize=ann_fs, xycoords='axes fraction', \
                        verticalalignment='top')
        # Axes params
        axs[a].invert_xaxis()
#         axs[a].grid(linestyle='-', linewidth='0.5', color='0.7')

    # Axes labels
        teff = r'log ($T_{\mathrm{eff}}$ /K)'
        axs[a].set_xlabel(teff)
    axs[0].set_ylabel(r'log ($L / \mathrm{L}_{\odot}$)')

    # Axes limits
    # axs[0].set_xlim(4.15,3.45)
    # axs[0].set_ylim(-0.5,3.5)

    # Colorbar
    f, eps = adjust_plot_readability(fig=f, fontOG=None, plot='isos')
    add_axes_list = [cb_right-eps[1]+0.005, cb_bot+eps[2], 0.04, cb_top+eps[3]-cb_bot-eps[2]]
                            # (pushes right, pushes up, width, height)
    cb_ax = f.add_axes(add_axes_list)
    cbar = get_isocbar(sm=p, cax=cb_ax, f=f, ticks=plot_times)

#     plt.tight_layout()
    if save is not None: plt.savefig(save)
    plt.show()

    adjust_plot_readability(fig=None, fontOG=fontOG)
    return None

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
            rows = cbdf.loc[midx,:] # get the correct rows from the df
#                     axs[a].scatter(mdf.log_Teff, mdf.log_L, color=cbc[cb], marker=mrkr, s=100)
            axs[a].scatter(rows.log_Teff, rows.log_L,
                           c=rows.log10_isochrone_age_yr, cmap=isocmap, vmin=isovmin, vmax=isovmax,
                           **kwargs)

        # Title panels
        if cb==0:
            lbl = r'NoDM'
        else:
            lbl = r'$\Gamma_B = 10^{}$'.format(cb)
        axs[a].annotate(lbl,(ann_lmarg,ann_tmarg), fontsize=ann_fs, xycoords='axes fraction', \
                        verticalalignment='top')
        # Axes params
        axs[a].invert_xaxis()
#         axs[a].grid(linestyle='-', linewidth='0.5', color='0.7')

    # Axes labels
        teff = r'log ($T_{\mathrm{eff}}$ /K)'
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
    print("YOU SHOULD UPDATE plot_isos() TO USE THE FUNCTION adjust_plot_readability().")
    cb_ax = f.add_axes(add_axes_list)
    cbar = get_isocbar(sm=p, cax=cb_ax, f=f)

#     plt.tight_layout()
    if save is not None: plt.savefig(save)
    plt.show()

# fe HR isochrones plot

# fs Hottest MS Teff
def get_hotT_data(data_dir=r2tf_dir+'/', age_range=(10**7.75,10**10.25), pdf=None):
    """ Loads data from all history.data files in data_dir subdirectories matching c[0-6].
            if pdf arg given it uses this instead of loading from scratch.
        Interpolates log_Teff to plot_times.

        pdf (df): as in pdf = get_profilesindex_df(profile_runs_dir=data_dir)

        Returns a df with columns [mass, cb, star_age, log_Teff].
    """

    plot_times = np.linspace(age_range[0],age_range[1],1000)
    plot_data = []
    cols = ['star_age','log_Teff', 'center_h1']

    # get df of [mass, cb, path_to_LOGS] for history files and itterate thru them
    pdf = get_profilesindex_df(profile_runs_dir=data_dir) if pdf is None else pdf
    hdf_dirs = pdf.groupby('path_to_LOGS').max() # .max() to get one row per star
    for i, (Ldir, row) in enumerate(hdf_dirs.iterrows()):

        # if row.mass < 4.5: continue
        # if row.cb > 1.0 and row.cb <5: continue

        print('Processing dir {}'.format(Ldir))
        # get the MS data from the history file
        hpath = os.path.join(Ldir,'history.data')
        df = pd.read_csv(hpath, header=4, sep='\s+', usecols=cols)
        df = cut_HR_hdf(df, cuts=['ZAMS','H-3']) # only use MS

        # add Teff interpolated to ages in plot_times
        pt = np.ma.masked_outside(plot_times,df.star_age.min(),df.star_age.max())
        pt = pt.data[pt.mask == False]
        dftmp = pd.DataFrame({'star_age':pt, 'log_Teff':np.nan})
        # set star_age as the index and concatenate the dfs
        df = df.loc[:,['star_age','log_Teff']].set_index('star_age')
        dftmp = dftmp.set_index('star_age')
        age_Teff = pd.concat([df,dftmp], axis=0).sort_index()
        # interpolate log_Teff
        age_Teff = age_Teff.interpolate(method='values')

        # create df with the needed info
        age_Teff = age_Teff.loc[pt,:].reset_index()
        age_Teff['mass'] = row.mass
        age_Teff['cboost'] = int(row.cb)
        plot_data.append(age_Teff)

    # concat the dfs and keep only the hottest star for a given age and cboost
    df = pd.concat(plot_data, axis=0, ignore_index=True)
    df = df.loc[df.groupby(['cboost','star_age']).log_Teff.idxmax(),:]

    return df

def plot_hottest_Teff(save=None, pdf=None, plot_data=hotTeff_csv):
    """ Plots highest log_Teff of any mass in MS
        pdf arg is passed to get_hotT_data(pdf=pdf) if plot_data is None
        plot_data (None, df, or str):
                        None: loads data from individual history.data files (very slow)
                        df: as given by plot_data = pf.get_hotT_data()
                        str: path to plot_data stored as csv
    """
    if save is None:
        plt.figure()
    else:
        plt.figure(figsize=(savefigw, savefigh*1.1))
    ax = plt.gca()

    # get the plot data (if plot_data arg not passed as df)
    if plot_data is None:
        plot_data = get_hotT_data(pdf=pdf)
    elif type(plot_data) == str:
        plot_data = pd.read_csv(plot_data)

    cbgroup = plot_data.groupby(('cboost'))
    for i, (cb,dat) in enumerate(cbgroup):
        cmapdict = get_cmapdict(cb,len(dat.index))
        zord = np.abs(cb-6)
        plt.scatter(dat.star_age.apply(np.log10), dat.log_Teff, s=3, **cmapdict, zorder=zord)

    cbar = get_cbcbar()

#     Axes
    # plt.semilogx()
    # ax.xaxis.set_major_locator(plt.MultipleLocator(1.0))
    # ax.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    # plt.minorticks_on()
    # ax.xaxis.set_minor_locator(plt.MultipleLocator(0.1))
    # ax.xaxis.set_minor_formatter(FormatStrFormatter(""))


    plt.xlabel(r'log ($Isochrone\ Age$ /yr)')
    plt.ylabel(r'log ($T_{\mathrm{eff}}$ /K)')
#     plt.ylabel(r'$\frac{\Delta \tau_{MS}}{\tau_{MS,\ Fid}}$')
#     plt.title('Change in Main Sequence Lifetimes')
    plt.tight_layout()
    if save is not None: plt.savefig(save)
    plt.show()

    return None

def plot_hottest_Teff_from_isos(isodf, cut_axes=True, save=None):
    if save is None:
        plt.figure()
        # mnum=5e2
    else:
        plt.figure(figsize=(savefigw, savefigh))
        # mnum=1e4
    ax = plt.gca()

    cbgroup = isodf.sort_values('log10_isochrone_age_yr').groupby(('cboost'))
    for i, (cb,dat) in enumerate(cbgroup):
        if cut_axes:
            dat = cut_HR_hdf(dat, cuts=['ZAMS', 'TAMS'])
            # cbdf0 = cut_HR_hdf(cbdf0, cuts=['ZAMS','TAMS'])

        # dat.set_index('log10_isochrone_age_yr', inplace=True, verify_integrity=True)
        # print(dat.sample())
        tmax = dat.groupby('log10_isochrone_age_yr')['log_Teff'].max()
        # print(len(idx), len(dat.log10_isochrone_age_yr[idx]))
        # print(tmax)
        # print(tmax.index)
        # print()
        cmapdict = get_cmapdict(cb,len(tmax))
        zord = np.abs(cb-6)
        plt.scatter(tmax.index, tmax, s=3, **cmapdict, zorder=zord)
        plt.plot(tmax.index, tmax, c=get_cmap_color(cb), zorder=zord)
        # plt.scatter(dat.log10_isochrone_age_yr[idx],dat.log_Teff[idx], s=5, **cmapdict)
        # plt.plot(dat.log10_isochrone_age_yr[idx],dat.log_Teff[idx], c=get_cmap_color(cb))

    # cbar = get_cbcbar()

#     Axes
    # plt.semilogx()
    # ax.xaxis.set_major_locator(plt.MultipleLocator(1.0))
    # ax.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    # plt.minorticks_on()
    # ax.xaxis.set_minor_locator(plt.MultipleLocator(0.1))
    # ax.xaxis.set_minor_formatter(FormatStrFormatter(""))


    plt.xlabel(r'log ($Isochrone\ Age$ /yr)')
    plt.ylabel(r'log ($T_{\mathrm{eff}}$ /K)')
#     plt.ylabel(r'$\frac{\Delta \tau_{MS}}{\tau_{MS,\ Fid}}$')
#     plt.title('Change in Main Sequence Lifetimes')
    plt.tight_layout()
    if save is not None: plt.savefig(save)
    plt.show()

    return None

# fe Hottest MS Teff

# fs 3.5 Msun profiles
# c0mods=[775,856,949]
# c6mods=[775,1557,1683]
def plot_m3p5(peeps=None, h1_legend=True, save=None):
    """ peeps = list of h1cuts (names defined in get_h1cuts()) to plot
    """

    cb = 6
    cbmods = get_h1_modnums(mass=3.5)
    c0mods = cbmods[0]
    c6mods = cbmods[cb]

    if peeps is None:
        peeps = list(get_h1cuts()[0].keys())
        print(peeps)

    cbc = [get_cmap_color(0, cmap=cbcmap), get_cmap_color(cb, cmap=cbcmap)] # plot colors

    # For annotations:
     # placement for extra heat arrows in axes fraction corrds:
    delth = 0.055; deltv = 0.15
    l = 0.94; r = l + delth; b = 0.70; t = b + deltv
    eps = 0.1
    xarws = [[(r,b),(l,t)],
             [(r,b),(l,t)],
             [(r,b-eps),(l,t-eps)],
             [(r,b-eps),(l,t-eps)],
             [(r,b-eps),(l,t-eps)],
             [(r,b-eps),(l,t-eps)],
             [(r,b-eps),(l,t-eps)],
             [(r,b-eps),(l,t-eps)],
             ]

    # Tick directions
    # mpl.rcParams['xtick.direction'] = 'out'
    # mpl.rcParams['ytick.direction'] = 'out'

    ntimes = len(peeps)
    nrows = 2
    ncols = 1

    f = plt.figure() if save is None else plt.figure(figsize=(savefigw_vert, savefigh_vert*nrows**2))
    outer = gs.GridSpec(nrows=ntimes,ncols=ncols, figure=f, hspace=0.35, right=0.83)
    inner = [gs.GridSpecFromSubplotSpec(nrows,ncols, subplot_spec=outer[o], hspace=0) \
                    for o in range(ntimes)]

    a = -1
    h1c_names, titles = get_h1cuts()
    for t, peep in enumerate(list(h1c_names.keys())):
        c0mod = c0mods[t]
        c6mod = c6mods[t]
        if peep not in peeps:
            continue
        elif math.isnan(c0mod) or math.isnan(c6mod):
            continue
        else:
            a = a+1 # iterate outer subplot

        axe = f.add_subplot(inner[a][0])
        axtwin = axe.twinx()
        if a != ntimes-1:
            axc = f.add_subplot(inner[a][1], sharex=axe)
        else:
            axc = f.add_subplot(inner[a][1]) # need the xtick labels to show up correctly

        df = [ get_pdf(0, c0mod), get_pdf(cb, c6mod) ] # list of dataframes
        for i, d in enumerate(df):
            # Plot nuclear energy (use arrays to avoid auto generating legend label):
            axe.plot(np.array(d.mass), np.array(d.eps_nuc_plus_nuc_neu), c=cbc[i])
            # Plot extra heat:
            axtwin.plot(d.mass, d.extra_heat, ls='-.', c=cbc[i])

            # Plot convection:
            xfrac_lbl = get_xfrac_lbl(d)
            log_D = np.log10(10**d.log_D_conv + 10**d.log_D_ovr)
            axc.plot(d.mass, log_D, c=cbc[i], label=xfrac_lbl)

        # Annotate with arrows:
        kwargs = {  'xycoords':"axes fraction", 'textcoords':"axes fraction",
                    'arrowprops':{  'arrowstyle':"->",
                                    'connectionstyle':"angle,angleA=90,angleB=0",
                                    'alpha':0.5 }
                }
        axe.annotate('',xarws[a][0], xytext=xarws[a][1], **kwargs)

        # axes limits
        axc.set_ylim(-1,19.9)
        xmax = 0.95
        axe.set_xlim(0,xmax)

        if h1_legend:
            axc.legend()
        # Labels and Titles
        axe.set_title(titles[peep])
        # axe.set_title(peep if (plt.rcParams["text.usetex"]==False) else r'$\textbf{'+peep+'}$')
        # SEE https://matplotlib.org/3.1.0/api/text_api.html#matplotlib.text.Text
        # FOR TEXT ADJUSTMENT OPTIONS (e.g. 'position' and 'rotation')
        axe.set_ylabel(r'$\ \ \ \epsilon_{\rm{nuc}}\ [\frac{\rm{erg}}{\rm{g\, s}}]$')
        # axe.tick_params(labelrotation=-45)
        axtwin.set_ylabel(r'$\ \ \ \epsilon_{\rm{DM}}\ [\frac{\rm{erg}}{\rm{g\, s}}]$')
        # axtwin.tick_params(labelrotation=45)
        axc.set_ylabel(r'log($D/ [\frac{\rm{cm}^2}{\rm{s}}]$)')
        # axc.tick_params(labelrotation=-45)
        axe.set_xticklabels([])
    axc.set_xlabel(r'$mass(<r)/\mathrm{M}_{\odot}$')
    f.suptitle(r'3.5 M$_{\odot}$ Profiles', fontsize=40)

    axc.set_xlim(0,xmax) # this axis is not connected to axe to avoid tick labels on axe

    if save is not None: plt.savefig(save)
    plt.show()

def plot_m3p5_sidebyside(peeps=None, save=None):
    """ peeps = list of h1cuts (names defined in get_h1cuts()) to plot
    """

    xmax = 0.95
    cb = 6

    cbmods = get_h1_modnums(mass=3.5)
    c0mods = cbmods[0]
    c6mods = cbmods[cb]

    if peeps is None:
        peeps = list(get_h1cuts()[0].keys())
    ntimes = len(peeps)

    cbc = [get_cmap_color(0, cmap=cbcmap), get_cmap_color(cb, cmap=cbcmap)] # plot colors

    # For annotations:
     # placement for extra heat arrows in axes fraction corrds:
    delth = 0.055; deltv = 0.1
    l = 0.93; r = l + delth; b = 0.80; t = b + deltv
    # r2 = l2 + delth; b2 = 0.80; t2 = b2 + deltv
    eps = 0.4
    # fr = 1.
    xarws = [[(r,b),(l,t)],
             [(r,b-0.05),(l,t)],
             [(r,b-0.05),(l,t)],
             [(r,b-0.05),(l,t)],
             [(r,b-0.05),(l,t)],
             [(r,b-0.2),(l,t)],
             [(r,b-0.2),(l,t)],
             # [(r,b),(l,t)],
             # [(r,b),(l,t)],
             # [(r,b),(l,t)],
             # [(r,b),(l,t)]
             ]
     # placement for nuclear burning arrows
    r = l; l = r - delth; b = 100; t = b + deltv
    narws = [
            # [(l,t),(r,b)],
            #  [(l,t),(r,b)],
            #  [(l,t),(r,b)],
            #  [(l,t),(r,b)],
            #  [(l,t),(r,b)],
            #  [(l,t),(r,b)],
            #  [(l,t),(r,b)],
            #  [(l,t),(r,b)]
             ]

    if save is None:
        f, axs = plt.subplots(nrows=ntimes,ncols=2, sharex=True)
    else:
        f, axs = plt.subplots(nrows=ntimes,ncols=2, sharex=True,
                              figsize=(savefigw, savefigh+2))
    axtwin = [axs[t,0].twinx() for t in range(ntimes)]

    a = -1
    h1c_names = list(get_h1cuts()[0].keys())
    for t, peep in enumerate(h1c_names):
        c0mod = c0mods[t]
        c6mod = c6mods[t]
        if peep not in peeps:
            continue
        elif math.isnan(c0mod) or math.isnan(c6mod):
            continue
        else:
            a = a+1 # iterate axs row

        df = [ get_pdf(0, c0mod), get_pdf(cb, c6mod) ] # list of dataframes

        for i, d in enumerate(df):
            d = d[d.mass<xmax] # slice the portion that will be visible so axes scale correctly

            # Plot nuclear energy (use arrays to avoid auto generating legend label):
            axs[a,0].plot(np.array(d.mass), np.array(d.eps_nuc_plus_nuc_neu), c=cbc[i])#, zorder=-i+6)

            # Plot mixing:
            xfrac_lbl = get_xfrac_lbl(d)
            log_D = np.log10(10**d.log_D_conv + 10**d.log_D_ovr)
            axs[a,1].plot(d.mass, log_D, c=cbc[i], label=xfrac_lbl)
#             axs[a,1].plot(d.mass, np.log10(d.np), c=cbc[i], label=xfrac_lbl)

        # Plot extra heat:
        axtwin[a].plot(df[1].mass, df[1].extra_heat, ls=':', c=cbc[i])

        axs[a,1].legend(loc='best') # show xfrac legend
        # Annotate with times:
        time_name = peep if (plt.rcParams["text.usetex"]==False) else r'$\textbf{'+peep+'}$'
        axs[a,0].annotate(time_name, (1.05,1.), xycoords='axes fraction',
                          annotation_clip=False, **{'fontweight':'bold'})

        # Set axis limits:
        axs[a,0].set_xlim(-0.0,xmax)
        axs[a,1].set_ylim(-1,17)

    # Annotate with arrows:
    kwargs = {'xycoords':"axes fraction", 'textcoords':"axes fraction",
    'arrowprops':{'arrowstyle':"->", 'connectionstyle':"angle,angleA=90,angleB=0", \
            'alpha':0.5}
            }
    axs[0,0].annotate('',xarws[0][0], xytext=xarws[0][1], **kwargs)
    # axs[a,0].annotate('',narws[t][0], xytext=narws[t][1], **kwargs)

    # Set energy legend:
    axs[0,0].plot(d.mass.iloc[0], d.eps_nuc_plus_nuc_neu.iloc[0], c='0.5', \
                    label=r'$\epsilon_{nuc}$ [erg/g/s]')
    axs[0,0].plot(d.mass.iloc[0], d.extra_heat.iloc[0], ls=':', c='0.5', \
                    label=r'$\epsilon_{\chi}$ [erg/g/s]')
    axs[0,0].legend(loc='center')

    # Set mixing legend:
    axs[0,1].plot(d.mass.iloc[0], d.log_D_conv.iloc[0], c='0.5',
                  label=r'Convective Mixing\\log(diff. coef [cm$^2$/s])')
    axs[0,1].legend(loc='center')


    # Labels and titles
    axs[a,0].set_xlabel(r'$mass(<r)/\mathrm{M}_{\odot}$')
    axs[a,1].set_xlabel(r'$mass(<r)/\mathrm{M}_{\odot}$')

    # Tick directions
    mpl.rcParams['xtick.direction'] = 'out'
    mpl.rcParams['ytick.direction'] = 'out'
    # plt.xticks(rotation=45)
    # plt.yticks(rotation=45)

#     plt.subplots_adjust(hspace = 0.5)
    plt.tight_layout()

    if save is not None: plt.savefig(save)
    plt.show()

# fe 3.5 Msun profiles

# fs 1.0 Msun c3 profiles
def plot_m1p0c3(peeps=None, h1_legend=True, fix_yscale=False, save=None):
    """ peeps = list of h1cuts (names defined in get_h1cuts()) to plot
    """

    xmax = 0.21
    cb = 3
    cbmods = get_h1_modnums(mass=1.0)
    c0mods = cbmods[0]
    c3mods = cbmods[cb]

    if peeps is None:
        peeps = list(get_h1cuts()[0].keys())
        print(peeps)

    cbc = [get_cmap_color(0, cmap=cbcmap), get_cmap_color(cb, cmap=cbcmap)] # plot colors

    # For annotations:
     # placement for extra heat arrows in axes fraction corrds:
    delth = 0.055; deltv = 0.15
    l = 0.94; r = l + delth; b = 0.75; t = b + deltv
    eps = 0.2
    xarws = [[(r,b),(l,t)],
             [(r,b-eps),(l,t-eps)],
             [(r,b-eps),(l,t-eps)],
             [(r,t-eps),(l,b-eps)],
             [(r,b-eps),(l,t-eps)],
             [(r,b-eps),(l,t-eps)],
             [(r,b-eps),(l,t-eps)],
             [(r,b-eps),(l,t-eps)],
             ]

    # Tick directions
    # mpl.rcParams['xtick.direction'] = 'out'
    # mpl.rcParams['ytick.direction'] = 'out'

    ntimes = len(peeps)
    nrows = 2
    ncols = 1

    f = plt.figure() if save is None else plt.figure(figsize=(savefigw_vert, savefigh_vert))
    outer = gs.GridSpec(nrows=ntimes,ncols=ncols, figure=f, hspace=0.35, right=0.83)
    inner = [gs.GridSpecFromSubplotSpec(nrows,ncols, subplot_spec=outer[o], hspace=0) \
                    for o in range(ntimes)]

    a = -1
    h1c_names, titles = get_h1cuts()
    for t, peep in enumerate(list(h1c_names.keys())):
        c0mod = c0mods[t]
        c3mod = c3mods[t]
        if peep not in peeps:
            continue
        elif math.isnan(c0mod) or math.isnan(c3mod):
            continue
        else:
            a = a+1 # iterate outer subplot

        axe = f.add_subplot(inner[a][0])
        axtwin = axe.twinx()
        if a != ntimes-1:
            axc = f.add_subplot(inner[a][1], sharex=axe)
        else:
            axc = f.add_subplot(inner[a][1]) # need the xtick labels to show up correctly

        df = [ get_pdf(0, c0mod, mass=1.0), get_pdf(cb, c3mod, mass=1.0) ] # list of dataframes
        for i, d in enumerate(df):
            d = d[d.mass < xmax+0.05] # cut data so y axis scales nicely

            # Plot nuclear energy (use arrays to avoid auto generating legend label):
            axe.plot(np.array(d.mass), np.array(d.eps_nuc_plus_nuc_neu), c=cbc[i])
            # Plot extra heat:
            axtwin.plot(d.mass, d.extra_heat, ls='-.', c=cbc[i])

            # Plot temperatures:
            xfrac_lbl = get_xfrac_lbl(d)
            axc.plot(np.array(d.mass), np.array(d.logT), c=cbc[i], label=xfrac_lbl)
            if i != 0:
                xls = {'ls':':', 'lw':1}
                axc.plot(np.array(d.mass), np.array(np.log10(d.wimp_temp)), c=cbc[i], **xls)


        # Annotate with arrows:
        kwargs = {  'xycoords':"axes fraction", 'textcoords':"axes fraction",
                    'arrowprops':{  'arrowstyle':"->",
                                    'connectionstyle':"angle,angleA=90,angleB=0",
                                    'alpha':0.5 }
                }
        axe.annotate('',xarws[a][0], xytext=xarws[a][1], **kwargs)

        # axes limits
        axe.set_xlim(0,xmax)
        if fix_yscale:
            axe.set_ylim(-1,55)
            axtwin.set_ylim(-25,10)
            axc.set_ylim(7.0,7.33)

        if h1_legend:
            axc.legend()
        # Labels and Titles
        axe.set_title(titles[peep])
        # axe.set_title(peep if (plt.rcParams["text.usetex"]==False) else r'$\textbf{'+peep+'}$')
        # SEE https://matplotlib.org/3.1.0/api/text_api.html#matplotlib.text.Text
        # FOR TEXT ADJUSTMENT OPTIONS (e.g. 'position' and 'rotation')
        axe.set_ylabel(r'$\ \ \ \epsilon_{\rm{nuc}}\ [\frac{\rm{erg}}{\rm{g\, s}}]$')
        # axe.tick_params(labelrotation=-45)
        axtwin.set_ylabel(r'$\ \ \ \epsilon_{\rm{DM}}\ [\frac{\rm{erg}}{\rm{g\, s}}]$')
        # axtwin.tick_params(labelrotation=45)
        axc.set_ylabel(r'log($T/ \rm{K}$)')
        # axc.tick_params(labelrotation=-45)
        axe.set_xticklabels([])
    axc.set_xlabel(r'$mass(<r)/\mathrm{M}_{\odot}$')
    f.suptitle(r'1.0 M$_{\odot}$ Profiles', fontsize=40)

    axc.set_xlim(0,xmax) # this axis is not connected to axe to avoid tick labels on axe

    if save is not None: plt.savefig(save)
    plt.show()

def plot_m1p0c3_sidebyside(peeps=None, save=None):
    """ cbmods = ordered list of lists of model numbers to plot
            as returned by get_h1_modnums()
        peeps = list of h1cuts (names defined in get_h1cuts()) to plot
    """
    xmax = 0.21
    cb = 3

    cbmods = get_h1_modnums(mass=1.0)
    c0mods = cbmods[0]
    c3mods = cbmods[cb]

    if peeps is None:
        peeps = list(get_h1cuts()[0].keys())
    ntimes = len(peeps)

    cbc = [get_cmap_color(0, cmap=cbcmap), get_cmap_color(cb, cmap=cbcmap)] # plot colors

    # For annotations:
     # placement for extra heat arrows:
    # delth = 0.015; deltv = 1.5
    # l = 0.19; r = l + delth; b = 20; t = b + deltv
    delth = 0.055; deltv = 0.1
    l = 0.93; r = l + delth; b = 0.80; t = b + deltv
    xarws = [[(r,b),(l,t)],
             [(r,b+10),(l,t+10)],
             [(r,b+35),(l,t+35)],
             [(r,b),(l,t)],
             [(r,b),(l,t)],
             [(r,b),(l,t)],
             [(r,b),(l,t)],
             [(r,b),(l,t)]]
    #  # placement for nuclear burning arrows
    # r = l; l = r - delth; b = 10; t = b + deltv
    # narws = [[(l,t),(r,b)],
    #          [(l,t),(r,b)],
    #          [(l,t),(r,b)],
    #          [(l,t),(r,b)],
    #          [(l,t),(r,b)],
    #          [(l,t),(r,b)],
    #          [(l,t),(r,b)],
    #          [(l,t),(r,b)]]

    if save is None:
        f, axs = plt.subplots(nrows=ntimes,ncols=2, sharex=True)
    else:
        f, axs = plt.subplots(nrows=ntimes,ncols=2, sharex=True,
                              figsize=(savefigw, savefigh+2))
    axtwin = [axs[t,0].twinx() for t in range(ntimes)]

    a = -1
    h1c_names = list(get_h1cuts()[0].keys())
    for t, peep in enumerate(h1c_names):
        if peep not in peeps:
            continue
        else:
            a = a+1
#             print(peep)

        df = [ get_pdf(0, c0mods[t], mass=1.0), get_pdf(cb, c3mods[t], mass=1.0) ] # list of dataframes

        for i, d in enumerate(df):
            d = d[d.mass<xmax] # slice the portion that will be visible so axes scale correctly
            # Plot nuclear energy (use arrays to avoid auto generating legend label):
            axs[a,0].plot(np.array(d.mass), np.array(d.eps_nuc_plus_nuc_neu), c=cbc[i])#, zorder=-i+6)
            # Plot Temperature:
            xfrac_lbl = get_xfrac_lbl(d)
            axs[a,1].plot(np.array(d.mass), np.array(d.logT), c=cbc[i], label=xfrac_lbl)
        # Plot wimp temp:
        xls = {'ls':'-.', 'lw':1}
        axs[a,1].plot(np.array(df[1].mass), np.array(np.log10(df[1].wimp_temp)), c=cbc[i], **xls)
        # Plot extra heat:
        axtwin[a].plot(df[1].mass, df[1].extra_heat, ls=':', c=cbc[i])

        axs[a,1].legend() # show xfrac legend
        # Annotate with times:
        time_name = peep if (plt.rcParams["text.usetex"]==False) else r'$\textbf{'+peep+'}$'
        axs[a,0].annotate(time_name, (1.05,1.), xycoords='axes fraction',
                          annotation_clip=False, **{'fontweight':'bold'})

        # Set axis limits:
        axs[a,0].set_xlim(-0.0,xmax)
        axs[a,1].set_xlim(-0.0,xmax)
#         axs[t,1].set_ylim(7.0,7.4)

    # Annotate with arrows:
    ap = {'arrowstyle':"->", 'connectionstyle':"angle,angleA=90,angleB=0", 'alpha':0.5}
    axs[0,0].annotate('',xarws[0][0], xytext=xarws[0][1], arrowprops=ap, xycoords='axes fraction')
    # axs[a,0].annotate('',narws[t][0], xytext=narws[t][1], arrowprops=ap)

    # Set energy legend:
    axs[0,0].plot(d.mass.iloc[0], d.eps_nuc_plus_nuc_neu.iloc[0], c='0.5', label=r'$\epsilon_{\mathrm{nuc}}$ [erg/g/s]')
    axs[0,0].plot(d.mass.iloc[0], d.extra_heat.iloc[0], ls=':', c='0.5', label=r'$\epsilon_{\mathrm{DM}}$ [erg/g/s]')
    axs[0,0].legend(loc='center')

    # Set temp legend:
#     axs[0,1].plot(d.mass.iloc[0], d.logT.iloc[0], c='0.5', label=r'log(T [K])')
#     axs[0,1].plot(d.mass.iloc[0], np.log10(d.wimp_temp.iloc[0]), c='0.5', **xls, label=r'log(T$_{\chi}$ [K])')
    axs[0,1].plot(-1, 7.1, c='0.5', label=r'log ($T$ [K])')
    axs[0,1].plot(-1, 7.1, c='0.5', **xls, label=r'log ($T_{\mathrm{DM}}$ [K])')
    axs[0,1].legend(loc='best')


    # Labels and titles
    axs[a,0].set_xlabel(r'$mass(<r)/\mathrm{M}_{\odot}$')
    axs[a,1].set_xlabel(r'$mass(<r)/\mathrm{M}_{\odot}$')

    # Tick directions
    mpl.rcParams['xtick.direction'] = 'out'
    mpl.rcParams['ytick.direction'] = 'out'

#     plt.subplots_adjust(hspace = 0.5)
    plt.tight_layout()

    if save is not None: plt.savefig(save)
    plt.show()
# fe 1.0 Msun c3 profiles

# fs 1.0 Msun c6 profiles
def plot_m1p0c6(plot_times=None, h1_legend=False, fix_yscale=False, save=None):
    """ peeps = list of h1cuts (names defined in get_h1cuts()) to plot
    """

    xmax = 0.21
    cb = 6

    if plot_times is None:
        plot_times = list(get_osc_modnums().keys())

    cbc = [get_cmap_color(0, cmap=cbcmap), get_cmap_color(cb, cmap=cbcmap)] # plot colors

    # For annotations:
     # placement for extra heat arrows in axes fraction corrds:
    delth = 0.055; deltv = 0.15
    l = 0.94; r = l + delth; b = 0.75; t = b + deltv
    eps = 0.2
    xarws = [[(r,b),(l,t)],
             [(r,b-eps),(l,t-eps)],
             [(r,b-eps),(l,t-eps)],
             [(r,t-eps),(l,b-eps)],
             [(r,b-eps),(l,t-eps)],
             [(r,b-eps),(l,t-eps)],
             [(r,b-eps),(l,t-eps)],
             [(r,b-eps),(l,t-eps)],
             ]

    # Tick directions
    # mpl.rcParams['xtick.direction'] = 'out'
    # mpl.rcParams['ytick.direction'] = 'out'

    ntimes = len(plot_times)
    nrows = 2
    ncols = 1

    f = plt.figure() if save is None else plt.figure(figsize=(savefigw_vert, savefigh_vert*ntimes))
    outer = gs.GridSpec(nrows=ntimes,ncols=ncols, figure=f, hspace=0.35, right=0.83)
    inner = [gs.GridSpecFromSubplotSpec(nrows,ncols, subplot_spec=outer[o], hspace=0) \
                    for o in range(ntimes)]

    a = -1
    oscmods = get_osc_modnums() # dict
    om_names = list(get_osc_modnums().keys())
    for t, tname in enumerate(om_names):
        if tname not in plot_times:
            continue
        else:
            a = a+1

        axe = f.add_subplot(inner[a][0])
        axtwin = axe.twinx()
        if a != ntimes-1:
            axc = f.add_subplot(inner[a][1], sharex=axe)
        else:
            axc = f.add_subplot(inner[a][1]) # need the xtick labels to show up correctly

        print(cb, oscmods[tname])
        print()
        d = get_pdf(cb, oscmods[tname], mass=1.0)
        d = d[d.mass<xmax+0.05] # slice the portion that will be visible so axes scale nicely

        # Plot nuclear energy (use arrays to avoid auto generating legend label):
        axe.plot(np.array(d.mass), np.array(d.eps_nuc_plus_nuc_neu), c=cbc[1])
        # Plot extra heat:
        xls = {'ls':'-.', 'lw':1}
        axtwin.plot(d.mass, d.extra_heat, **xls, c=cbc[1])
        axtwin.axhline(0, c=cbc[0], **xls)

        # Plot temperatures:
        xfrac_lbl = get_xfrac_lbl(d)
        axc.plot(np.array(d.mass), np.array(d.logT), c=cbc[1], label=xfrac_lbl)
        xls = {'ls':':', 'lw':1}
        axc.plot(np.array(d.mass), np.array(np.log10(d.wimp_temp)), c=cbc[1], **xls)


        # Annotate with arrows:
        kwargs = {  'xycoords':"axes fraction", 'textcoords':"axes fraction",
                    'arrowprops':{  'arrowstyle':"->",
                                    'connectionstyle':"angle,angleA=90,angleB=0",
                                    'alpha':0.5 }
                }
        # axe.annotate('',xarws[a][0], xytext=xarws[a][1], **kwargs)

        # axes limits
        axe.set_xlim(0,xmax)
        if fix_yscale:
            if a != ntimes-1:
                axe.set_ylim(0,32)
                axtwin.set_ylim(-225,445)
                axc.set_ylim(7.03,7.11)
            else: # bottom axis (degen)
                axe.autoscale(axis='y', tight=False)
                axtwin.autoscale(axis='y', tight=False)
                axc.autoscale(axis='y', tight=False)

        if h1_legend:
            axc.legend()
        # Labels and Titles
        axe.set_title(tname)
        # axe.set_title(peep if (plt.rcParams["text.usetex"]==False) else r'$\textbf{'+peep+'}$')
        # SEE https://matplotlib.org/3.1.0/api/text_api.html#matplotlib.text.Text
        # FOR TEXT ADJUSTMENT OPTIONS (e.g. 'position' and 'rotation')
        axe.set_ylabel(r'$\ \ \ \epsilon_{\rm{nuc}}\ [\frac{\rm{erg}}{\rm{g\, s}}]$')
        # axe.tick_params(labelrotation=-45)
        axtwin.set_ylabel(r'$\ \ \ \epsilon_{\rm{DM}}\ [\frac{\rm{erg}}{\rm{g\, s}}]$')
        # axtwin.tick_params(labelrotation=45)
        axc.set_ylabel(r'log($T/ \rm{K}$)')
        # axc.tick_params(labelrotation=-45)
        axe.set_xticklabels([])
    axc.set_xlabel(r'$mass(<r)/\mathrm{M}_{\odot}$')
    f.suptitle(r'1.0 M$_{\odot}$ Profiles', fontsize=40)

    axc.set_xlim(0,xmax) # this axis is not connected to axe to avoid tick labels on axe

    if save is not None: plt.savefig(save)
    plt.show()

def plot_m1p0c6_sidebyside(plot_times=None, save=None):
    """ cbmods = ordered list of lists of model numbers to plot
            as returned by get_h1_modnums()
        plot_times = list of oscilation times (names defined in get_osc_modnums()) to plot
    """
    xmax = 0.25
    cb = 6

#     plot_times=None
    if plot_times is None:
        plot_times = list(get_osc_modnums().keys())
    ntimes = len(plot_times)

#     cbc = get_cmap_color(cb, cmap=cbcmap) # plot colors
    cl = 'maroon' # left axis color
    cr = 'darkblue' # right axis color

    # For annotations:
     # placement for extra heat arrows:
    delth = 0.015; deltv = 1.5
    l = 0.19; r = l + delth; b = 20; t = b + deltv
    xarws = [[(r,b),(l,t)],
             [(r,b+10),(l,t+10)],
             [(r,b+35),(l,t+35)],
             [(r,b),(l,t)],
             [(r,b),(l,t)],
             [(r,b),(l,t)],
             [(r,b),(l,t)],
             [(r,b),(l,t)]]
     # placement for nuclear burning arrows
    r = l; l = r - delth; b = 10; t = b + deltv
    narws = [[(l,t),(r,b)],
             [(l,t),(r,b)],
             [(l,t),(r,b)],
             [(l,t),(r,b)],
             [(l,t),(r,b)],
             [(l,t),(r,b)],
             [(l,t),(r,b)],
             [(l,t),(r,b)]]

    ncols = 2
    if save is None:
        f, axs = plt.subplots(nrows=ntimes,ncols=ncols, sharex=True)
    else:
        f, axs = plt.subplots(nrows=ntimes,ncols=ncols, sharex=True,
                                                          figsize=(savefigw, savefigh+6))
    axtwin = [ [axs[t,0].twinx() for t in range(ntimes)],
                [axs[t,1].twinx() for t in range(ntimes)] ]

    a = -1
    oscmods = get_osc_modnums() # dict
    om_names = list(get_osc_modnums().keys())
    for t, tname in enumerate(om_names):
        if tname not in plot_times:
            continue
        else:
            a = a+1

        print(cb, oscmods[tname])
        print()
        d = get_pdf(cb, oscmods[tname], mass=1.0)
        d = d[d.mass<xmax] # slice the portion that will be visible so axes scale nicely

        # Plot nuclear energy (use arrays to avoid auto generating legend label):
        axs[a,0].plot(np.array(d.mass), np.array(d.eps_nuc_plus_nuc_neu), c=cl)#, zorder=-i+6)

        # Plot Temperature:
        xfrac_lbl = get_xfrac_lbl(d)
#         tmtx = (10**d.logT - d.wimp_temp)/ 1e7
#         axs[a,1].plot(np.array(d.mass), np.array(tmtx), c=cbc)#, label=xfrac_lbl)
        axs[a,1].plot(np.array(d.mass), np.array(d.logT), c=cl)#, label=xfrac_lbl)
#         axs[a,1].legend() # show xfrac legend

        # Plot logRho:
        axtwin[1][a].plot(np.array(d.mass), np.array(d.logRho), c=cr)

        # Plot wimp temp:
        xls = {'ls':':', 'lw':1}
#         axs[a,1].plot(np.array(d.mass), 0*np.array(d.mass), c=cbc, **xls)
        axs[a,1].plot(np.array(d.mass), np.array(np.log10(d.wimp_temp)), c=cl, **xls)

        # Plot extra heat:
        axtwin[0][a].plot(d.mass, d.extra_heat, c=cr)
        axtwin[0][a].plot(d.mass, 0*d.mass, c=cr, lw=0.5) # horizontal line at 0

        # Set axis colors:
        for itmp in range(ncols):
            axs[a,itmp].tick_params(axis='y', colors=cl)
#         axs[a,1].spines['left'].set_color(cl)
            axtwin[itmp][a].tick_params(axis='y', colors=cr)



        # Annotate with times:
        time_name = tname if (plt.rcParams["text.usetex"]==False) else r'$\textbf{'+tname+'}$'
        axs[a,0].annotate(time_name, (1.05,1.), xycoords='axes fraction',
                          annotation_clip=False, **{'fontweight':'bold'})

#         # Annotate with arrows:
#         ap = {'arrowstyle':"->", 'connectionstyle':"angle,angleA=90,angleB=0", 'alpha':0.5}
#         axs[a,0].annotate('',xarws[t][0], xytext=xarws[t][1], arrowprops=ap)
#         axs[a,0].annotate('',narws[t][0], xytext=narws[t][1], arrowprops=ap)

    # Set axis limits:
        axs[a,0].set_ylim(0,32)
        axtwin[0][a].set_ylim(-225,445)
        axs[a,1].set_ylim(7.03,7.11)
        axtwin[1][a].set_ylim(1.6,2.45)
    # set bottom axes limits
    for itmp in range(ncols):
#         axs[a,itmp].relim()
        axs[a,itmp].autoscale(axis='y', tight=False)
        axtwin[itmp][a].autoscale(axis='y', tight=False)
    axs[a,0].set_xlim(-0.0,xmax)
    axs[a,1].set_xlim(-0.0,xmax)

    # Set energy legend:
    axs[1,0].plot(d.mass.iloc[0], d.eps_nuc_plus_nuc_neu.iloc[0], c=cl, label=r'$\epsilon_{nuc}$ [erg/g/s]')
    axs[1,0].plot(d.mass.iloc[0], d.extra_heat.iloc[0], c=cr, label=r'$\epsilon_{\chi}$ [erg/g/s]')
    axs[1,0].legend(loc='upper right')

    # Set temp/rho legend:
    axs[1,1].plot(-1, 7.1, c=cl, label=r'log($T$ [K])')
    axs[1,1].plot(-1, 7.1, c=cl, **xls, label=r'log($T_{\mathrm{DM}}$ [K])')
    axs[1,1].plot(-1, 7.1, c=cr, label=r'log(Rho [g/cm3])')
    axs[1,1].legend(loc='upper right')


    # Labels and titles
    axs[a,0].set_xlabel(r'$mass(<r)/\mathrm{M}_{\odot}$')
    axs[a,1].set_xlabel(r'$mass(<r)/\mathrm{M}_{\odot}$')

    # Tick directions
    mpl.rcParams['xtick.direction'] = 'out'
    mpl.rcParams['ytick.direction'] = 'out'

#     plt.subplots_adjust(top = 2.0)
    plt.tight_layout()

    if save is not None: plt.savefig(save)
    plt.show()
# fe 1.0 Msun c6 profiles

# fs 1.0 Msun c6 Kippenhahn
def plot_m1p0c6_kipp(plot_times=None, from_file=False, save=None):
    """ Gets and plots the history.data file.
        from_file should be list [c0,c6] of specific dirs (parent of LOGS dir)
                    to get correct history.data with burning cols.
    """

    cut_axes = True
    if plot_times is None:
        plot_times = list(get_osc_modnums().keys())
    ntimes = len(plot_times)

    # xlim = (1.9*10**8, 4.*10**8)
    xlim = (1.6*10**8, 3.7*10**8)
    ylim = (0.,0.7)
    ylimtwin = (-0.7, 0.5)
    cb = 6
    burn_zone_colors = ['w', 'darkorange','darkred']

    ff = from_file if from_file==False else from_file[0]
    hdf0 = get_hdf(0, mass=1.0, from_file=ff)
    ff = from_file if from_file==False else from_file[1]
    hdf = get_hdf(cb, mass=1.0, from_file=ff)

    if cut_axes:
        hdf = cut_HR_hdf(hdf) # cut preMS
        hdf.star_age = hdf.star_age - hdf.star_age.iloc[0] # set age to start at ZAMS
        hdf0 = cut_HR_hdf(hdf0)
        hdf0.star_age = hdf0.star_age - hdf0.star_age.iloc[0]

    if save is None:
        plt.figure()
    else:
        plt.figure(figsize=(savefigw, savefigh+2))
    ax = plt.gca()
    axtwin = ax.twinx()


    # Plot burning:
    burn_cols0 = get_burn_cols(hdf0)
    bzones0 = burn_cols0.groupby('burn_type', sort=True)
    burn_cols = get_burn_cols(hdf)
    bzones = burn_cols.groupby('burn_type', sort=True)
    for b, (bz, cols) in enumerate(bzones):
        c = burn_zone_colors[bz]
        alph = 0.7
#         icols = interp_burn_cols(cols, xlim) # SLOW WITH MINIMAL EFFECT
        icols=cols
        ax.fill_between(icols.star_age, icols.mass_bottom, y2=icols.mass_top,
                        step='mid', color=c, alpha=alph)
        cols0 = bzones0.get_group(bz).sort_values('star_age')
#         icols0 = interp_burn_cols(cols0)
        ax.plot(cols0.star_age, cols0.mass_top, color=c)


    # Plot osc times vlines:
    oscmods = get_osc_modnums() # dict
    om_names = list(get_osc_modnums().keys())
    # For osc times annotations:
    anny = ylim[1] - 0.05
    r = -90
    tdlta = 5e6
    fs = 10
    for t, tname in enumerate(om_names):
        if tname not in plot_times: continue
        tmod = oscmods[tname]
        tage = hdf[hdf.model_number==tmod].star_age.values[0]
        plt.axvline(tage, lw=0.5, c='k')
        # Osc times annotations:
        rot = -r if tname == 'Time5' else r
        td = 1.5*tdlta if tname=='Degen' else (0 if tname=='Time5' else tdlta)
        ax.annotate(tname,(tage-td,anny), rotation=rot, fontsize=fs)

    # Plot L
    Lc = 'b'
    axtwin.plot(hdf.star_age, hdf.log_L, color=Lc)
    axtwin.plot(hdf.star_age, hdf.log_LH, color=Lc, ls=':')
    axtwin.plot(hdf.star_age, hdf.log_R, color=Lc, ls='-.')
    axtwin.set_ylim(ylimtwin)
    axtwin.tick_params(axis='y', colors=Lc)
    axtwin.set_ylabel(r'log ($L/ \mathrm{L}_{\odot}$)', color=Lc)


    # Axes params:
    ax.semilogx()
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel(r'$Stellar Age$ [yr]')
    ax.set_ylabel(r'$mass(<r)\ [\mathrm{M}_{\odot}]$')

    mpl.rcParams['xtick.direction'] = 'out'
    mpl.rcParams['ytick.direction'] = 'out'

    plt.tight_layout()
    if save is not None: plt.savefig(save)
    plt.show()

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

# fe 1.0 Msun c6 Kippenhahn⁄
