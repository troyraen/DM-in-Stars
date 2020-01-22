#!/bin/bash

# generate rnMESA#.sh files for condor
# mkdirs
# write inlists
# run condor

# script input variables
mkcln="${1:-1}" #  = 1 will execute ./mk and ./clean
# hard coded variables
maindir="/home/tjr63/mesaruns"
RUNS="RUNS_final1"
masses=$(seq 5.00 -.05 .80)
prcsn=2 # mass string/float precision
cboost=$(seq 0 6)


### MAIN PROGRAM
### generates files and runs mesa models using Condor

cd ${maindir}
if [ "${mkcln}" = 1 ]; then
        ./clean
        ./mk
fi

source bash_scripts/write_inlists
for cb in "${cboost[@]}"; do
    for mass in "${masses[@]}"; do
        mass=$( printf "%.${prcsn}f" $mass ) # force precision
        mstr="m${mass:0:1}p${mass:2:${prcsn}}"
        RUN="${maindir}/${RUNS}/c${cb}/${mstr}"
        mkdir -p "${RUN}/LOGS" "${RUN}/png" "${RUN}/photos"
        chmod -R 766 ${RUN}
        write_inlists ${maindir} ${RUN} ${mass} ${cb}
    done
done

for n in 'seq 0 9'; do
    for mass in 'seq $n 10 ${#masses[@]}'
