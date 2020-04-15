- [Create a DMS Conda env on Osiris](#condaenv)
- [Create derived data files](#deriveddata)
    - [`history_pruned.data`](#prunehist)
    - [`descDF.csv`](#descdf)
    - [`hotTeff.csv`](#hotT)
    - [Do some checks](#checks)
- [Create Raen2020 paper plots](#makeplots)
    - [Changes made to `RUNS_2test_final` (MESA-r10398) plot code](#r10398Changes)

<a name="condaenv"></a>
# Create a DMS Conda env on Osiris
```bash
conda create -n DMS python=3.7 numpy pandas ipython matplotlib astropy scipy
```

<a name="deriveddata"></a>
# Create derived data files
<!-- fs  -->

<a name="prunehist"></a>
## Create `history_pruned.data` files
<!-- fs -->

Prune `history.data` files according to `data_proc.prune_hist()`

To use the pruned files in future code, set `usepruned=True` at the top of each file
    - `gen_descDF_csv.py`
    - `plot_fncs.py`

```python
from pathlib import Path
import data_proc as dp
import plot_fncs as pf

# iterate over all model dirs and prune the `history.data` file, creating `history_pruned.data`
currently_running = [(1.40,5), (1.20,5), (1.00,5)] # (mass, cb), skip these
rootPath = Path(pf.datadir)
for massdir, mass, cb in dp.iter_starModel_dirs(rootPath):
    if (mass,cb) in currently_running: continue
    LOGSdir = massdir / 'LOGS'
    dp.prune_hist(LOGSdir, skip_exists=False, ow_exists=True)

# add prune by variable changes
LOGSdir = Path(pf.datadir + '/c6/m2p00/LOGS')
dp.prune_hist(LOGSdir, skip_exists=False, ow_exists=True)

# check a c6 model
hpath = LOGSdir / 'history.data'
hdf = dp.load_history(hpath)
hpath = LOGSdir / 'history_pruned.data'
hdfprun = dp.load_history(hpath)

plt.figure()
ax = plt.gca()
for i, (lbl, h) in enumerate({'full': hdf, 'pruned': hdfprun}.items()):
    colors = ['g','b']
    quargs= {   'label':lbl, 'ax':ax, 'kind':'scatter', 'alpha':0.5,
                'color':colors[i]
                }
    h.plot('log_Teff','log_L', **quargs)
plt.title(f'{LOGSdir}')
plt.tight_layout()
plt.savefig(pf.plotdir + '/checkprun_HR.png')
# This looks good
```
<!-- fe ## Create `history_pruned.data` files -->


<a name="descdf"></a>
## Create `descDF.csv`
<!-- fs -->
Staring with old code in `../../Glue/data_proc.py` and moving necessary items to `gen_descDF_csv.py` and `data_proc.py`.

__Install `py_mesa_reader`__ so this works:
```bash
# Install `mesa_reader`
conda activate DMS
cd DMS
git clone git@github.com:wmwolf/py_mesa_reader.git
cd py_mesa_reader
pip install .
# pip uninstall mesa_reader
```

Once the above is completed (including [pruning history files](#prunehist)), __generate `descDF.csv`__:
```bash
cd ~/DMS/mesaruns_analysis/_Paper/figures
# WARNING: this uses `history_pruned.data` files only
python gen_descDF_csv.py
```

<!-- Need columns:
    - [ ]  Index: `star_index`
    - [ ]  cboost
    - [ ]  mass
    - [ ]  masscc_avg
    - [ ]  MStau
    - [ ]  TACHeB_model -->
<!-- fe ## Create `descDF.csv` -->


<a name="hotT"></a>
## Create `hotTeff.csv`
<!-- fs -->
```python
%run plot_fncs
# get the data
hotT_data = get_hotT_data(data_dir=datadir+'/', age_range=(10**7.75,10**10.25))
# write the file
hotT_data.to_csv(datadir+'/hotTeff.csv')
```
<!-- fe ## Create `hotTeff.csv` -->


<a name="checks"></a>
## Do some checks
<!-- fs -->
```python
# check that profiles with priorities have saved the right events
rootPath = Path(pf.datadir)
hdflst = [  dp.load_history(massdir / 'LOGS/history.data', (mass,cb)) \
            for massdir, mass, cb in dp.iter_starModel_dirs(rootPath) if cb<3
            ]
hdf = pd.concat(hdflst)

# find offset in number of models between profile saved and event in hdf
pidf = pf.pidxdfOG.set_index(['mass','cb'])
dfprior = pd.DataFrame(columns=['ZAMS','lAMS','TAMS','TACHeB'], index=hdf.index.unique())
for h in hdflst:
    idx = h.index.unique()
    p = pidf.loc[idx,['priority','model_number']]

    zams_hdf = h.loc[h.center_h1<(h.center_h1.iloc[0]-0.0015),'model_number'].min()
    zams_pidf = p.loc[p.priority==99,'model_number']
    dfprior.loc[idx,'ZAMS'] = zams_hdf - zams_pidf

    lams_hdf = h.loc[h.center_h1<1e-3,'model_number'].min()
    lams_pidf = p.loc[p.priority==95,'model_number']
    dfprior.loc[idx,'lAMS'] = lams_hdf - lams_pidf

    tams_hdf = h.loc[h.center_h1<1e-12,'model_number'].min()
    tams_pidf = p.loc[p.priority==94,'model_number']
    dfprior.loc[idx,'TAMS'] = tams_hdf - tams_pidf

    he_hdf = h.loc[h.center_he4<1e-3,'model_number'].min()
    print(he_hdf)
    he_pidf = p.loc[p.priority==92,'model_number']
    dfprior.loc[idx,'TACHeB'] = he_hdf - he_pidf
dfprior.fillna(0, inplace=True)

# plot histograms
dfprior.hist()
plt.suptitle('histograms of hdf model# - pidf model#')
plt.tight_layout()
plt.savefig(pf.plotdir+'/checkprioritys_hist.png')
```

<img src="temp/checkprioritys_hist.png" alt="/checkprioritys_hist" width=""/>
ZAMS difference likely just due to using different definition in run_star_extras.
Need to be aware of TACHeB.

<!-- fe ## Do some checks -->

<!-- fe # Create derived data files -->


<a name="makeplots"></a>
# Create plots for Raen2020 paper
<!-- fs -->

<!-- fs plots -->
- Osiris
- `defDM` branch
- `home/tjr63/DMS/mesaruns_analysis/_Paper/figures/` (Osiris) directory


### setup and testing
```python
%run plot_fncs
pidf = pidxdfOG # df of profiles.index files
cb, mass = 0, 1.0
modnum = pidf.loc[((pidf.mass==mass)&(pidf.cb==cb)&(pidf.priority==97)),'model_number'].iloc[0]
# hdf = load_hist_from_file(0, mass=1.0, from_file=True, pidxdf=pidf) # 1p0c0 history df
hdf = get_hdf(cb, mass=mass) # single history df
pdf = get_pdf(cb, modnum, mass=mass, rtrn='df') # single profile df
```


### delta MS Tau
```python
descdf = get_descdf(fin=fdesc)
save = [None, plotdir + '/MStau.png', finalplotdir + '/MStau.png']
plot_delta_tau(descdf, cctrans_frac='default', which='avg', save=save[1])
```

- [ ]  check unpruned history.data fro m2.55c4 to see if this is causing the spike

Note that there is a problem in matplotlib version 3.1.3
when trying to use a colormap with a scatter plot and data of length 1
See https://github.com/matplotlib/matplotlib/issues/10365/
I fixed this in `plot_delta_tau()` and other fncs below by doing
`plt.scatter(np.reshape(x,-1), np.reshape(y,-1), c=np.reshape(c,-1))`


### Teff v Age
```python
mlist = [1.0, 2.0, 3.5,]# ,0.8, 5.0]
cblist = [4, 6]
from_file = [False, get_r2tf_LOGS_dirs(masses=mlist, cbs=cblist+[0])]
                    # Only need to send this dict once.
                    # It stores history dfs in dict hdfs (unless overwritten)
save = [None, plotdir+'/Teff.png', finalplotdir+'/Teff.png']
plot_Teff(mlist=mlist, cblist=cblist, from_file=from_file[1], descdf=descdf, save=save[1])
```

- [ ]  start ages = 0 at ZAMS
- [ ]  why does lifetime difference in 1Msun look bigger than in 2Msun (contradicting MStau plot)?


### HR Tracks
```python
# mlist = [0.8, 1.0, 2.0, 3.5, 5.0]
# cblist = [4, 6]
from_file = [False, True, get_r2tf_LOGS_dirs(masses=mlist, cbs=cblist+[0])]
                        # Only need to send this dict once.
                        # It stores history dfs in dict hdfs (unless overwritten)
save = [None, plotdir+'/tracks.png', finalplotdir+'/tracks.png']
plot_HR_tracks(mlist=mlist, cblist=cblist, from_file=from_file[0], descdf=descdf,
                  save=save[1])
```

- [ ]  why is there a jaunt in the NoDM leave MS line?
- [ ]  remove pre-ZAMS portion


### Isochrones
```python
isodf = load_isos_from_file(fin=iso_csv, cols=None)
isoages = get_iso_ages(isodf)
plot_times = [age for i,age in enumerate(isoages) if i%5==0][3:]
print(plot_times)
# plot_times = [8.284, 8.4124, 8.8618, 9.1828, 9.4396, 9.6964, 9.9532, 10.017400000000002]
# plot_times = [7.0, 7.3852, 7.642, 7.8346, 8.0272, 8.155599999999998]
cb = [4,6]
for c in cb:
    save = [None, plotdir+'/isos_cb'+str(c)+'_symb.png', \
            finalplotdir+'/isos_cb'+str(c)+'.png']
    plot_isos_ind(isodf, plot_times=plot_times, cb=c, save=save[1])
```


### Hottest MS Teff
```python
save = [None, plotdir+'/hotTeff.png', finalplotdir+'/hotTeff.png']
plot_hottest_Teff(plot_data=hotTeff_csv, save=save[1], resid=False)
```

- [ ]  rerun when all models have completed


### 3.5 Msun profiles
```python
# cbmods = get_h1_modnums(mass=3.5)
# print(cbmods)
peeps = [ 'ZAMS', 'IAMS', 'H-3', 'H-4' ]
save = [None, plotdir+'/m3p5.png', finalplotdir+'/m3p5.png']
h1_legend = [False, True]
plot_m3p5(peeps=peeps, h1_legend=h1_legend[1], save=save[1])
```

- [ ]  last two profiles look like they are at the wrong times


### 1.0 Msun profiles
```python

```

<!-- fe plots -->


<a name="r10398Changes"></a>
## Changes made to `RUNS_2test_final` (plots-r10398) plot code
<!-- fs -->
- [x]  update paths to files and directories

<!-- fe ## Changes -->
<!-- fe # Create plots for Raen2020 paper -->
