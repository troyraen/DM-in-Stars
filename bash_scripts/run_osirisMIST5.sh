#!/bin/bash

### runs mesa models with specified params
run_key=$1
maindir="/home/tjr63/DMS/mesaruns"
cd ${maindir}
RUNS="RUNS_runSettings/${run_key}"

cp "inlist_master${run_key}" "inlist_master"
echo "copied inlist_master${run_key} to inlist_master"
if [[ "${run_key}" =~ ^(_mist01m9|_mist03m9|_mist06m9)$ ]]; then
    cp "src/run_star_extras${run_key}.f" "src/run_star_extras.f"
    echo "copied src/run_star_extras${run_key}.f to src/run_star_extras.f"
else
    cp "src/run_star_extras_default_plus_DM.f" "src/run_star_extras.f"
    echo "copied src/run_star_extras_default_plus_DM.f to src/run_star_extras.f"
fi

./clean
./mk

source ${maindir}/bash_scripts/do_mesa_run.sh

declare -A mvals=( [m0p8]=0.8 [m1p0]=1.0 [m1p25]=1.25 [m1p5]=1.5 [m1p75]=1.75 [m2p0]=2.0 [m2p5]=2.5 [m3p0]=3.0 [m3p5]=3.5 [m4p0]=4.0 [m4p5]=4.5 [m5p0]=5.0 )


declare -a mord=( m0p8 )
for cb in 5; do
    for mass in "${mord[@]}"; do
        do_mesa_run "${maindir}" "${RUNS}/c${cb}/${mass}" "${mvals[${mass}]}" "${cb}" 0 "master" 0
    done
done
