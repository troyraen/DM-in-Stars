__This file continued in new branch__ `runSettings`

- [new Main Runs](#mainruns)


# Questions

- [ ]  Why do some runs not finish? e.g. m4p5c0 (and many others)
    - [ ]  Need to review inlist options. Currently set to match MIST as much as possible, but several things had to be removed and the remaining are still complicated and I don't understand them all.

- [ ]  Runtimes
    - [ ]  Given that MS lifetimes results are different and the runs are taking a lot longer, need to decide how many models to re-run.

    - [ ]  check/fix inlist first. Once models are finishing, try to reduce run time
        - [ ]  possibly alter mesh, see options [here](https://lists.mesastar.org/pipermail/mesa-users/2011-September/000526.html)



<a name="mainruns"></a>
# Start new Main Runs in dir `RUNS_3test_final`
<!-- fs  -->
On Osiris node3, using
```
# inlist:
use_dedt_form_of_energy_eqn = .true. # first runs had this commented out but m4p5 c0 failed
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

__m4p5c0 (and several others) still did not finish... check MIST inlist stuff__

<!-- fe # Start new Main Runs -->
