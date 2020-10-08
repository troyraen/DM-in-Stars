# CURRENTLY DOING:


Getting profiles:
<!-- fs -->

- [x] ./bash_scripts/profile_run.sh 6 "m1p00" 1.0 725 40
- [-] need c6 models 1669 - 1825 to get complete oscillation cycle
- [x] ./bash_scripts/profile_run.sh 6 "m1p00" 1.0 1830 200
- [-] need c6 models 1900 (1907) - 1957 to get begin degeneracy and IAMS
- [x] ./bash_scripts/profile_run.sh 6 "m1p00" 1.0 1965 75
- [-] need to get burning regions in history.data (change history_columns.list)
- [ ] ./bash_scripts/profile_run.sh 6 "m1p00" 1.0 2000 5
- [ ] ./bash_scripts/profile_run.sh 0 "m1p00" 1.0 1000 5
- [ ]
- [ ]
- [x] ./bash_scripts/profile_run.sh 0 "m1p00" 1.0 725 50
- [x] ./bash_scripts/profile_run.sh 0 "m1p00" 1.0 865 150
- [x] ./bash_scripts/profile_run.sh 3 "m1p00" 1.0 725 40
- [x] ./bash_scripts/profile_run.sh 3 "m1p00" 1.0 1 5
- [x] ./bash_scripts/profile_run.sh 3 "m1p00" 1.0 1851 40
- [x] ./bash_scripts/profile_run.sh 3 "m1p00" 1.0 2785 140
- [ ]
- [x] ./bash_scripts/profile_run.sh 0 "m3p50" 3.5 949 200
- [x] ./bash_scripts/profile_run.sh 6 "m3p50" 3.5 775 5
- [x] ./bash_scripts/profile_run.sh 6 "m3p50" 3.5 1683 200
- [ ]
<!-- fe Getting profiles: -->


# log onto Osiris
cd mesaruns
git pull master
## two options:
## 1. Hand Pick Params
Open new screen and use ./bash_scripts/run_osiris.sh
to run a batch of hand-picked masses and cboosts.
(Make sure this script has correct inputs first.)
e.g.: ./bash_scripts/run_osiris.sh 0
## 2. Bulk Run with Screen Spawn
Use bash_scripts/bmr_caller (or bash_scripts/bulk_mesa_run.sh individually)
to generate sequences of masses and cboosts
and automatically spawn screen processes to run batches of them
## either way
(to look at std.out use:) sed "/\.png\/png/d" LOGS/STD.out | tail -n 100 | less

add option to make isochrones

# from Roy
download using bash_scripts/dwnld_from_osiris.sh
create glue csv files using Glue/data_proc_README.py
then use datadir/Glue/README.py (in separate Glue project directory) for plotting

# isochrones
open git project iso
use readme and readme_glue

))
# Archive Currently Doing


Set in the last line of the file ./bash_scripts/bmr_caller.sh.
Then call with ./bash_scripts/bmr_caller.sh
- [x] bmr_caller "RUNS_2test_final" 5.0 -0.05 0.79 5 1 6 4
- [x] bmr_caller "RUNS_2test_final" 5.0 -0.05 0.79 0 1 2 4 1
- [x] bmr_caller "RUNS_2test_final" 5.0 -0.05 0.79 3 1 4 4 1
- [ ]
- [x] bmr_caller "RUNS_2test_final" 1.93 -0.1 0.8 0 1 6 4 1
- [x] bmr_caller "RUNS_2test_final" 1.98 -0.1 0.8 0 1 6 4 1
- [x] bmr_caller "RUNS_2test_final" 2.38 -0.05 2.01 4 1 4 4 1
- [x] bmr_caller "RUNS_2test_final" 3.38 -0.05 2.11 5 1 5 4 1
- [x] bmr_caller "RUNS_2test_final" 4.98 -0.05 2.79 6 1 6 4 1
- [ ] bmr_caller "RUNS_2test_final" 4.98 -0.05 1.92 0 1 0 4 1
- [ ]
- [ ]

save history every step
run through WD?
mass increment 0.05
