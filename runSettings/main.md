- [new Main Runs after resolving largeLH problem/branch](#firstruns)

- [Cleanup/fix inlist and run_star_extras to better match MIST](#fixMIST)



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
In dir `RUNS_3test_final`.
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
- [x]  Copy newest default files over.
- [x]  Update defaults to match my needs for added DMS stuff and my preferences fo non-physics settings (e.g., saving history columns, profiles, etc.)
- [ ]  Update to MIST settings (see Table 1 of MIST1 paper (Choi16) and dir `MIST_MESA_files`)
    - [ ]  

<!-- fe # Cleanup/fix inlist and run_star_extras to better match MIST -->
