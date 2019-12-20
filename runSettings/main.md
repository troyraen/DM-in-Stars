- [new Main Runs after resolving largeLH problem/branch](#firstruns)

- [Cleanup/fix inlist and run_star_extras to better match MIST](#fixMIST)
    - [Baseline run using MESA m1p0 inlist plus settings needed for DM](#defDM)
    - [Run with full MIST options](#fullMISTmf)


# Questions

- [ ]  __Why do some runs not finish?__ e.g. m4p5c0 (and many others)
    - [ ]  Need to review inlist options. Currently set to match MIST as much as possible, but several things had to be removed and the remaining are still complicated and I don't understand them all.

- [ ]  Runtimes
    - [ ]  Given that MS lifetimes results are different and the runs are taking a lot longer, need to decide how many models to re-run.

    - [ ]  check/fix MIST stuff first. Once models are finishing, try to reduce run time
        - [ ]  possibly alter mesh, see options [here](https://lists.mesastar.org/pipermail/mesa-users/2011-September/000526.html)


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

```python
%run fncs
hdf = load_all_history(dr=dr, run_key='')
h = hdf.drop(index=('',6,1.0))
```

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


### <a name="fullMISTmf">__run key: \_fullMISTmf (mf: minus files, does not include MIST `MESA_files`)__</a>
LOGS in dir `RUNS_runSettings/fullMIST` since `_defDM m1p0c6` still running.
On Osiris node2

```bash
./clean
./mk

nohup nice ./bash_scripts/run_osiris1.sh &>> STD1_nohup_MIST.out &
```

__These models are not running well. C0 models taking a _very_ long time and some (m2p5c0) quit early due to `dt < min_timestep_limit`.__


### To do next:
- [ ]  Check:
    - [ ]  do all runs finish?
    - [ ]  are nx, np negative?
    - [ ]  Compare:
            - runtimes, # models, # backups, # retries, # newton iters (avg/step), avg dt/step
            - MStau, Tc, Rhoc, wimp_temp, Teff(hottest), L(end of MS),
    - [ ]  m2p5c6 "root must be bracketed"

- [ ]  Try with:
    - [ ]  default energy scheme
    - [ ]  no rotation
    - [ ]  smaller net

<!-- fe # Cleanup/fix inlist and run_star_extras to better match MIST -->


### Sand

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
