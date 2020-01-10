#!/bin/bash

run_key=$1
ont=$2 # OMP_NUM_THREADS
maindir="/home/tjr63/DMS/mesaruns"
# bashdir="${maindir}/bash_scripts"

cd ${maindir}
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


export OMP_NUM_THREADS=${ont}

nohup nice ./bash_scripts/run_osirisRTT.sh ${run_key} ${ont} "1" &>> STD1_nohup_RTT.out &
nohup nice ./bash_scripts/run_osirisRTT.sh ${run_key} ${ont} "2" &>> STD2_nohup_RTT.out &
nohup nice ./bash_scripts/run_osirisRTT.sh ${run_key} ${ont} "3" &>> STD3_nohup_RTT.out &
