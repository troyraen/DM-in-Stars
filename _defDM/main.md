# Final runs using `defDM` settings
See [runSettings branch](https://github.com/troyraen/DM-in-Stars/blob/runSettings/runSettings/main.md) for details.

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

## Generate bash scripts and do the runs
- [x]  create run_osiris.sh script that cylces through mass and cboost
- [ ]  start the runs

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
