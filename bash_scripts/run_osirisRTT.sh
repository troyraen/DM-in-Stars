#!/bin/bash

### runs mesa models with specified params
run_key=$1
ont=$2 # needed to put ont runs in different dirs
sim=$3 # needed to put simultaneous runs in different dirs
maindir="/home/tjr63/DMS/mesaruns"
cd ${maindir}
RUNS="RUNS_runtimeTests/${run_key}/threads${ont}/sim${sim}"


declare -A mvals=( [m0p8]=0.8 [m1p0]=1.0 [m1p25]=1.25 [m1p5]=1.5 )
declare -a mord=( m1p25 )


source ${maindir}/bash_scripts/do_mesa_run.sh

for cb in 6; do
    for mass in "${mord[@]}"; do
        do_mesa_run "${maindir}" "${RUNS}/c${cb}/${mass}" "${mvals[${mass}]}" "${cb}" 0 "master" 0
    done
done
