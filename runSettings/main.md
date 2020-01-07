- [new Main Runs after resolving largeLH problem/branch](#firstruns)

- [Cleanup/fix inlist and run_star_extras to better match MIST](#fixMIST)
    - [Baseline run using MESA m1p0 inlist plus settings needed for DM](#defDM)
    - [Run with full MIST options](#fullMISTmf)


# Questions

- [x]  __Why do some runs not finish?__ e.g. m4p5c0 (and many others). Need to review inlist options. Currently set to match MIST as much as possible, but several things had to be removed and the remaining are still complicated and I don't understand them all. _Updated to incorporate MESA-r12115 inlists and run_star_extras.f_

- [ ]  Runtimes
    - [ ]  Given that MS lifetimes results are different and the runs are taking a lot longer, need to decide how many models to re-run.

    - [ ]  check/fix MIST stuff first. Once models are finishing, try to reduce run time
        - [ ]  possibly alter mesh, see options [here](https://lists.mesastar.org/pipermail/mesa-users/2011-September/000526.html)

- [ ]  Check long run that just finished...
    - [ ]  which settings did it use?
    - [ ]  what was the runtime?
    - [ ]  what is delta tau_MS?

- [ ]  Which settings to use?
    - [ ]  can I get mist02m9 to complete runs?
    - [ ]  what is min log dt limit set to?
    - [ ]  what are the differences between mist02m9 and defDM?
        - [ ]  do they make a difference in evolution (tau_MS, convection stuff) or runtime?


-----------------------------------------------------------------------------
<a name="firstruns"></a>
# new Main Runs after resolving largeLH problem/branch
<!-- fs  -->
LOGS in dir `RUNS_3test_final`.
On Osiris node3, using
```
# inlist:
use_dedt_form_of_energy_eqn = .true. # very first runs had this commented out but m4p5c0 failed
use_gold_tolerances = .true.
# Runs dir from run_osiris#.sh:
RUNS="RUNS_3test_final"
```

```bash
./clean
./mk

nohup nice ./bash_scripts/run_osiris1.sh &>> STD1_nohup.out &
nohup nice ./bash_scripts/run_osiris2.sh &>> STD2_nohup.out &
```

__m4p5c0 (and several others) still did not finish.__ Terminate with errors (see STD.out in LOGS dir):.
```
stopping because of problems dt < min_timestep_limit
terminated evolution: cannot find acceptable model
termination code: min_timestep_limit
```
__Check MIST inlist stuff.__

<!-- fe # new Main Runs after resolving largeLH problem/branch -->


-----------------------------------------------------------------------------
<a name="fixMIST"></a>
# Cleanup/fix inlist and run_star_extras to better match MIST
<!-- fs -->

<!-- fs defDM -->
Starting with newest default files for `MESA-r12115`, `test_suite/1M_pre_ms_to_wd`:
`inlist_to_end_agb` (using some necessary settings from `inlist_start`) and `src/run_star_extras.f`.

- [x]  Rename the `inlist_master` and `src/run_star_extras.f` (append `_ow_12172019`) files I have been using for easy reference to previous settings.
- [x]  Copy newest MESA default files over.
- [x]  Update defaults to match my needs for added DMS stuff and my preferences fo non-physics settings (e.g., saving history columns, profiles, etc.)
- [x]  Test these settings (before incorporating MIST) to establish a baseline and see if c0 models will finish. Using file versions `inlist_master_default_plus_DM` and `run_star_extras_default_plus_DM`:

### <a name="defDM">__run key: \_defDM__</a>
LOGS in dir `RUNS_runSettings`
On Osiris node3

```bash
./clean
./mk

nohup nice ./bash_scripts/run_osiris1.sh &>> STD1_nohup.out &
```

__These models ran fine, though m1p0c6 model is taking a long time (currently ~2 days and has not yet hit IAMS)__
<!-- fe defDM -->

<!-- fs fullMISTmf -->
- [x]  Update to MIST settings (see Table 1 of MIST1 paper (Choi16) and dir `MIST_MESA_files`)
    - [x]  Update both files (everything except high mass stars (>10Msun) options)
        - [x]  Wind schemes have changed in new MESA version. See [controls defaults](http://mesa.sourceforge.net/controls_defaults.html) under `cool_wind_AGB_scheme` for info/instructions. Amounts to changing `Reimers_wind_eta -> Reimers_scaling_factor` and `Blocker_wind_eta -> Blocker_scaling_factor` in both files.
    - [-]  Replace MESA files with those in MIST_MESA_files/mesa_source and do clean, mk, export (see MIST_MESA_files/README)
        - [ ]  In `mesa-r12115/atm/private/e2_pairs.dek` line 460 change `E2_NPAIRS` -> `NPAIRS` _not working because multiple files call this one_
        - [-]  In `mesa-r12115/atm/public/atm_def.f90` add line 143: `integer, parameter :: E2_NPAIRS = npairs`
        - [-]  In `mesa-r12115/atm/private/table_atm.f90` fix 2 `Warning: Nonconforming tab character`
        <!-- - Check: `mesa-r12115/atm/private/table_atm.f90`, line 344 `which_atm_option == atm_photosphere_tables` -->
        - [-]  Copy/paste `subroutine table_atm_shutdown()` from `table_atm_OG.f90` to `table_atm.f90`
            - [-]  Comment out line 194 `call free_table_summary(ai_db_wd_25)`
        - dir `atm` still not compiling, moving on...
        - dir `const` compiled fine
        - doing `./mk` in `mesa-r12115/star` gives `No rule to make target 'opacities.mod', needed by 'star_lib.o'`
        - __Unable to get these files into compatibility with mesa-r12115. Trying without these files.__ Most have to do with `atm` (boundary conditions), will just use MESA photosphere table option. `mesa_49.net` adds some isos and reactions that I don't _think_ will be important.


### <a name="fullMISTmf">__run key: \_fullMISTmf (mf: minus files, does not include MIST `MESA_files/mesa_source` files)__</a>
LOGS in dir `RUNS_runSettings/fullMIST` since `_defDM m1p0c6` still running.
On Osiris node2.

```bash
./clean
./mk

nohup nice ./bash_scripts/run_osiris1.sh &>> STD1_nohup_MIST.out &
```

__These models are not running well. C0 models taking a _very_ long time and m2p5c0 quit early due to `dt < min_timestep_limit`.__ I stopped them after ~24 hours (and m1p0c0 still hadn't finished).

__Next, start with the most basic/simple MIST inlist and run_star_extras options and then add options one at a time.__
<!-- fe fullMISTmf -->


### <a name="MISToptions">__MIST options__</a>
The following runs will have run keys coded with `\_mist#` where `#` in:

0.  Basic options: run_star_extras_default_plus_DM and inlist_master_default_plus_DM + solar abundances, eos, opacity, jina reaction rates, radiation turbulence, mass loss (AGB, RGB scaling factors). _run_key = _basicMIST == _mist0_
0.  `0w`: everything in 0 plus winds section, also add `other_wind` to `run_star_extras`.
1.  mesh and timestep params: varcontrol_target and mesh_delta_coeff options
2.  convection, mlt, semiconvection, thermohaline, and overshoot
3.  rotation
4.  mesa_49.net
5.  atm_option ~~+ winds~~
6.  diffusion
7.  `m7`: opacity (subtract from basic)
8.  `m8`: radiation turbulence (subtract from basic)
9.  `m9`: mass loss (subtract from basic)
Did not do postAGB burning (run_star_extras)

### <a name="basicMIST">__run key: \_basicMIST__</a>

\_basicMIST m1p5c0 would not finish, `log_dt_yr` down to -11 after MS. Tried subtracting 7, 8, 9 from 0. Only `m9` would finish m1p5c0. Adding winds controls to inlist and run_star_extras (`0w`) and testing... This did not work (m1p5c0 does not finish). Now start with `m9` and add 1-6 one at a time.

`\_mist01m9` ran overnight. m1p5c0 finished but m1p0c0 did not.

`\_mist02m9` had some c0 models that didn't finish, quit with dt < limit.

Trying 0m9 again to make sure it's running right... looks fine. Checked that mist0#m9 files (inlist and rse) match mist0m9. Continuing tests.

`\_mist03m9` c0 didn't finish.

`_mist04m9/` m1p5c0 didn't finish.

`_mist05m9/` m1p5c0 didn't finish.

`_mist06m9/` m1p0c0 didn't finish.

`_mist07m9/` I stopped it at m1p5c6... no such inlist file. this run had `inlist_master_mist06m9` and `run_star_extras_default_plus_DM.f`

```bash
nohup nice ./bash_scripts/run_osirisMIST.sh "_mist08m9" &>> STD1_nohup_MIST.out &
```


## Compare runs
```python
%run fncs
rk = ['defDM','mist0m9','mist01m9','mist02m9','mist03m9','mist04m9','mist05m9'
        ,'mist06m9','mist07m9']
hdf, pidf, rcdf = load_all_data(dr=dr, run_key=rk, get_history=False)

rk = ['defDM','mist0m9','mist02m9','mist06m9','mist07m9']
hdf, pidf, rcdf = load_all_data(dr=dr, run_key=rk, get_history=True)
# pd.options.display.max_columns = len(rcdf.columns)
# this one hasn't finished yet so can't change dir name directly:
# rcdf.drop(index=('',6,1.0), inplace=True)
plot_pidf(pidf, save=None)
plot_rcdf_finished(rcdf, save=None)
cols = ['runtime', 'steps', 'log_dt_min', 'end_priority', ]
plot_rcdf(rcdf, save=None, cols=cols)
cols = ['runtime', 'retries', 'backups', 'steps', 'log_dt_min']
plot_rcdf(rcdf, save=None, cols=cols)
cols = ['log_max_rel_energy_error', 'log_cum_rel_energy_error']
plot_rcdf(rcdf, save=None, cols=cols)
cols = ['end_priority', 'center_h1_end', 'center_he4_end',]
plot_rcdf(rcdf, save=None, cols=cols)

# look at different slices
h = hdf.loc[(),:]
hunfin = hdf.loc[hdf.finished==False,:]
# punfin = pidf.loc[pidf.finished==False,:]
runfin = rcdf.loc[rcdf.finished==False,:]
# idx = pd.IndexSlice
r2 = rcdf.loc[idx['mist02m9',0:6,0:5],:]
h2 = hdf.loc[idx['mist02m9',0:6,0:5],:]

plot_rcdf_finished(rcdf, save=None)
cols = ['runtime', 'steps', 'log_dt_min', 'end_priority', 'termCode', ]
plot_rcdf(r2, save=None, cols=cols)
cols = ['center_h1_end','center_he4_end']
plot_rcdf(r2, save=None, cols=cols)
r2.plot('star_age','log_dt', logx=True)

# sand: get controls file
dir = os.path.join(dr,'c6','m2p5_defDM')
cpath = os.path.join(dir,'LOGS/controls1.data')
cs = load_controls(cpath)

%run fncs
rk = ['defDM','mist0m9','mist02m9']
hdf, pidf, cdf, rcdf = load_all_data(dr=dr, run_key=rk, get_history=True)
rk = ['test']
hdf0, pidf0, cdf0, rcdf0 = load_all_data(dr=dr, run_key=rk, get_history=True)

cdf = pd.concat([cdf,cdf0])
c = control_diff(cdf.loc[idx[['test','defDM'],6,1.0],:])

############ sand
drop_cols = ['TRACE_HISTORY_VALUE_NAME', 'STAR_HISTORY_DBL_FORMAT',
             'STAR_HISTORY_INT_FORMAT', 'STAR_HISTORY_TXT_FORMAT',
             'PROFILE_INT_FORMAT', 'PROFILE_TXT_FORMAT', 'PROFILE_DBL_FORMAT',
             'FORMAT_FOR_FGONG_DATA', 'FORMAT_FOR_OSC_DATA'
            ]
parse_cols = ['MESH_LOGX_MIN_FOR_EXTRA', 'MESH_DLOGX_DLOGP_EXTRA',
              'MESH_DLOGX_DLOGP_FULL_ON', 'MESH_DLOGX_DLOGP_FULL_OFF',
              'XA_FUNCTION_SPECIES', 'XA_FUNCTION_WEIGHT', 'XA_FUNCTION_PARAM',
              'DIFFUSION_CLASS_REPRESENTATIVE', 'DIFFUSION_CLASS_A_MAX'
              ]
c = cdf.copy()
for col in c.columns:
    c[col] = c[col].apply(lambda x: ','.join(x) if type(x)==list else x)
    # print(col)
    unq = len(c[col].unique())
    if unq!=1:
        print(unq)
        print()

def tlist(itm):
    if type(itm)==list:
        itm = ','.join(itm)

for col in c.columns:
    if col in drop_cols: continue
    if col in parse_cols:
        c[col] = c[col].apply(lambda x: ','.join(x))
        # print('joined')
    print(col)
    unq = len(c[col].unique())
    if unq!=1:
        print(col)
        print(unq)
        print()

for col in c.columns:
    if col in drop_cols: continue
    if col in parse_cols:
        c[col] = c[col].apply(lambda x: ','.join(x))
        print('joined')
        print(col)
        print(len(c[col].unique()))
        print()


```


## To do next:
- [ ]  Check:
    - [ ]  do all runs finish?
    - [ ]  are nx, np negative?
    - [ ]  Compare:
            - runtimes, # models, # backups, # retries, # total newton iters, avg dt/step, min dt
            - MStau, Tc, Rhoc, wimp_temp, Teff(hottest), L(end of MS),
    - [ ]  m2p5c6 "root must be bracketed"

- [ ]  Try with:
    - [ ]  default energy scheme

<!-- fe # Cleanup/fix inlist and run_star_extras to better match MIST -->


## Sand:
<!-- fs Sand -->
creating debug_df:
```python
%run fncs
dir = os.path.join(dr,'c6','m4p5')
with open(os.path.join(dir,'LOGS/STD.out')) as fin:
    for line in reversed(fin.readlines()):
        try:
            if line.split()[0] == 'runtime':
                cols,vals = line.strip().replace('steps','steps:').split(':')
                cols = cols.strip().split(',')
                vals = vals.split()
                dbs = pd.Series(data=vals,index=cols)
                print(dbs)
                break
        except:
            pass


hdf, rcdf = load_all_history(dr=dr+'/fullMIST', run_key='')
```
<!-- fe Sand -->
