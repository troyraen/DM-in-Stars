# Cleanup repo for use by others
3/31/2020

Travis Hurst's student, Logan, is trying to use my code and having problems getting it running. Looking at the repo, I'm sure it's very confusing to anyone but me, so I'm cleaning it up. Will delete unnecessary files, create simple inlists for the `master` branch, and move all files that others don't need to use (e.g. bash scripts to run bulk jobs with my parameters) into a different branch.

## Changes to `master` branch
- [x]  Merge `defDM` branch changes (mostly for the upgrade to MESA-r12115) into `master`
- [x]  rename 'wimp' -> 'DM'
- [x]  delete paper and abstract submodules
- [x]  create new branch `master-tjraen` from `master` (with above changes completed)
- [x]  strip down `run_star_extras.f`
- [x]  history and profile lists -> current MESA-r12115 default files
- [x]  `inlist_master` -> `inlist` and strip down
- [ ]  `inlist_options_tmplt` -> `inlist_DM` and put all DM stuff here
    - [ ]  add options to set DM mass and xsection in this inlist
- [x]  update main `readme.md`
- [x]  delete unnecessary files and directories
- [x]  clean up `DM_module.f`
- [ ]  check that `master` branch works by running (in `sand` dir)
    - [ ]  1 Msun, c0
    - [ ]  1 Msun, c6

Remove lines from `sand/STD.out` for readability:
```python
bad_words = [' is_slope_negative returns false. ', ' retry     ']
with open('STD.out') as oldfile, open('STDshort.out', 'w') as newfile:
    for line in oldfile:
        if not any(bad_word in line for bad_word in bad_words):
            newfile.write(line)
```

Check on models
```python
# run this from within the _defDM dir to use its fncs.py
%run fncs
hpath = '/home/tjr63/sand/LOGS/history.data'
figpath = '/home/tjr63/sand/LOGS/HR.png'
# hdf = load_history(hpath) # doesn't work because `wimp_temp` column has been renamed
hdf = pd.read_csv(hpath, header=4, sep='\s+')
hdf['cb'], hdf['mass'] = 0, 1.0
hdf.set_index(['cb','mass'], inplace=True)
plot_HR(hdf, color='dt', title=None, save=figpath)
```

c0 model is taking a very long time to finish, but I think it's running fine. It finished the MS by model 350. First time lg_dt_yr goes negative is after H_cntr=0.



- [x]  check `RUNS_defDM` `STD.out` files for `WRITE` output from `DM_module.f` indicating that a suitable root for Tx could not be found. Look through files indicated in `prob.mods` to make sure the "problem model" timesteps were retried by MESA.

```bash
cd DMS/mesaruns/RUNS_defDM
find . -type f -print | xargs grep "problem model " &>> prob.mods
```


## Pull in changes to `master-tjraen`
- [ ]  history and profile lists from `defDM` (don't know why these didn't get copied over when I merged `defDM` to `master`)
- [ ]  pull in code cleanup changes that I made to `master`


## Brett's feedback
- [ ]  shorten repo intro (top line) to ~10 words
- [ ]  don't bold first part of readme
- [ ]  add a license, see [Marvin's](https://github.com/sdss/marvin/blob/master/LICENSE.md)
- [ ]  add a changelog, see [Marvin's](https://github.com/sdss/marvin/blob/master/CHANGELOG.rst)
- [ ]  name `master-tjraen` branch after the paper, `raen2020`
