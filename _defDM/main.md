# Final runs using `defDM` settings
See [runSettings branch](https://github.com/troyraen/DM-in-Stars/blob/runSettings/runSettings/main.md) for details.

<!-- fs -->
## Branch cleanup
- [x]  copy `run_star_extras_default_plus_DM.f` -> `run_star_extras.f`
- [x]  copy `inlist_master_default_plus_DM` -> `inlist_master`
- [x]  delete the unneeded versions of these files (copied from `runSettings` branch)


## Check that history_columns.list and profile_columns.list have all needed variables
- [x]  Variables needed for [MIST](https://github.com/aarondotter/iso)
- [x]  history cols needed for `final_plots/plot_fncs.py` (check calls to hdf and isodf)
- [x]  profile cols needed for `final_plots/plot_fncs.py` (check calls to pdf)
- [x]  extra: {energy conservation, }

_Not saving `burning_regions`, needed for plot_m1p0c6_kipp... this run (m1p0c6) is already done and takes ~20 days... if decide this is needed, will need to do another run._


## Copy `defDM` runs from `runSettings` branch runs to new dir in prep for filling in mass grid for final runs.

```bash
maindir="/home/tjr63/DMS/mesaruns"
cd ${maindir}
fromd="RUNS_runSettings"
tod="RUNS_defDM"

for c in $(seq 0 6); do
    todir="${tod}/c${c}"
    mkdir ${todir}
    for dir in $(ls -d ${fromd}/c${c}/*_defDM/); do
        cp -r ${dir} ${todir}/.
    done
done
```
<!-- fe -->

## Generate bash scripts and do the runs
<!-- fs -->
- [x]  Create run_osiris.sh script that cylces through mass and cboost
- [x]  Start the runs

```bash
# do 2 on Osiris wnode3 and 1 on wnode2

ssho3 # log on to wnode3
mr # cd to mesaruns
git fetch
git checkout defDM
git pull

nohup nice ./bash_scripts/run_osiris.sh 1 &>> STD_nohup_defDM1.out & # start with last mord
# wait for the first run to start. then:
nohup nice ./bash_scripts/run_osiris.sh 2 &>> STD_nohup_defDM2.out & # start with 2nd to last mord
# check that the first run started. then:
exit

ssho2 # log on to wnode2
mr # cd to mesaruns
nohup nice ./bash_scripts/run_osiris.sh 3 &>> STD_nohup_defDM3.out & # start with 3rd to last mord
# check that the first run started
exit
```
<!-- fe ## Generate bash scripts and do the runs -->


## Check runtimes
<!-- fs -->
```python
hdf, pi_df, c_df, rcdf = load_all_data(dr=dr, get_history=False)
plot_runtimes(rcdf, save='runtimes_Feb11.png')
```

<img src="runtimes_Feb11.png" alt="runtimes_Feb11" width=""/>

__Currently running models:__
<img src="unfinished_models_021120.png" alt="unfinished_models_021120" width=""/>

I think the three with `termCode = -1` are stuck in very small timesteps. Should check again in a few days and cancel them if they're still running.
<!-- fe ## Check runtimes -->

## Investigate models stuck in small timesteps
<!-- fs -->
Need to reduc history file sizes so can load reasonably.
This takes too long on Roy.
I could not install python3 on Osiris, check_for_reduc will not run with python2.
I could not log into Osiris from Korriban (don't have password).
Doing this manually in bash:

On Osiris:
```bash
cd DMS/mesaruns/bash_scripts/
./data_reduc.sh /home/tjr63/DMS/mesaruns/RUNS_defDM/c1/m1p15/LOGS
./data_reduc.sh /home/tjr63/DMS/mesaruns/RUNS_defDM/c1/m1p20/LOGS
./data_reduc.sh /home/tjr63/DMS/mesaruns/RUNS_defDM/c2/m1p10/LOGS
```

Clone new version of repo on Osiris for analyzing `mesaruns` data:
```bash
cd DMS
git clone git@github.com:troyraen/DM-in-Stars.git mesaruns_analysis
```

__Load data and see what's done:__
Doing this on Osiris in `mesaruns_analysis` dir.
```python
# Problem models
mods = ['m0p90c0','m2p55c1','m1p60c2','m2p65c2', # stopped due to `min_timestep_limit`
        'm1p05c1'] # still running 2/25
# Load data
hdf, pidf, cdf, rcdf = load_all_data(dr=dr, get_history=True, skip=mods)
# see what's done
rcdf.reset_index().plot('mass','cb',kind='scatter')
plt.show(block=False)
plot_runtimes(rcdf, save='runtimes_Feb25.png')

# Get problem models
mods = ['m0p90c0','m2p55c1','m1p60c2','m2p65c2', # stopped due to `min_timestep_limit`
        'm1p05c1'] # still running 2/25
hdfm, pidfm, cdfm, rcdfm = load_all_data(dr=dr, get_history=True, mods=mods)
reddtdf, probTxdf = load_dt_root_errors(dr=dr, mods=mods)

# histograms of log_dt
hdf.hist(by=['cb','mass'], column='log_dt', bins=25)
plt.savefig('plots/probmods_logdt_hist.png')
plt.show(block=False)

# histograms of reduce dt codes
reddtdf.code.apply(pd.value_counts).plot(kind='bar',by=['cb','mass'],subplots=True)
plt.savefig('plots/probmods_reddt_hist.png')
plt.show(block=False)

title = 'models that quit due to min_timestep_limit', color='dt'
plot_HR(mtlhdf, color='dt', title=title, save='plots/probmods_HR.png')

```

<img src="runtimes_Feb25.png" alt="runtimes_Feb25" width="400"/>

<img src="plots/probmods_logdt_hist.png" alt="plots/probmods_logdt_hist" width="400"/><img src="plots/probmods_reddt_hist.png" alt="plots/probmods_reddt_hist" width="400"/>

<img src="plots/probmods_HR.png" alt="plots/probmods_HR" width="400"/>





__Models that stopped early due to `min_timestep_limit`__
```python
# Models that quit early due to `min_timestep_limit`
mintlim = rcdf.loc[rcdf.termCode=='min_timestep_limit',:]
mods = ['m0p90c0','m2p55c1','m1p60c2','m2p65c2']
mtlhdf, __, __, __ = load_all_data(dr=dr, get_history=True, mods=mods)
# mtlh = hdf.loc[mintlim.index,:]
title = 'models that quit due to min_timestep_limit', color='dt'
plot_HR(mtlhdf, color='dt', title=title, save='HR_mintlim.png')

#
# plt.figure(figsize=(6,10))
# # ax = plt.gca()
# cols = ['end_priority','log_dt_min','center_h1_end','center_he4_end']
# mintlim.plot(y=cols,kind='bar',subplots=True)
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.show(block=False)
#
#
# notdone = rcdf.loc[rcdf.finished==False,:]
# ndh = hdf.loc[notdone.index,:]
#
# slow = rcdf.loc[rcdf.log_dt_min<-5,:]
# slh = hdf.loc[slow.index,:]
# plot_log_dt(slh)
#
# ndsl = hdf.loc[notdone.index.intersection(slow.index),:] # not done + low min timestep
#
current = rcdf.loc[rcdf.termCode==-1,:]
crh = hdf.loc[current.index,:]
title = 'currently running models 2/21/20'
plot_HR(crh, title=title, save='HR_current_022120.png')
current[cols]
#
# plot_HR(ndsl, save=None)


rcdf.log_max_rel_energy_error.sort_values()
```

<img src="HR_mintlim.png" alt="HR_mintlim" width=""/>
<img src="rcdf_mintlim_Feb21.png" alt="rcdf_mintlim_Feb21" width=""/>

<img src="HR_current_022120.png" alt="HR_current_022120" width=""/>
<img src="rcdf_current_Feb21.png" alt="rcdf_current_Feb21" width=""/>
Note m1p1c2 model HR looks funny because using history_reduc.data.

STD.out warnings (reduce dt and Tx set to Tcenter)
```python
mods = ['m0p90c0','m2p55c1','m1p60c2','m2p65c2', # mintlim
        'm1p05c1','m1p10c2','m0p85c4'] # current
reddtdf, probTxdf = load_dt_root_errors(dr=dr, mods=mods)
plot_reddt(reddtdf, title=None, save=None)

hdf, pi_df, c_df, rcdf = load_all_data(dr=dr, get_history=True, mods=mods)
```

<!-- fe ## Investigate models stuck in small timesteps -->
