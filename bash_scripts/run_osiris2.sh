#!/bin/bash

### runs mesa models with specified params
maindir="/home/tjr63/DMS/mesaruns"
cd ${maindir}
RUNS="RUNS_3test_final"

# Ask user, run make/clean?
echo
echo "Run clean and make in dir ${maindir}?"
echo "  0 = no; 1 = yes"
read mkcln
if [ "${mkcln}" = 1 ]; then
        ./clean
        ./mk
fi

declare -A mvals=( [m0p9]=0.9 [m1p2]=1.2 [m1p7]=1.7 [m2p2]=2.2 [m2p7]=2.7 [m3p2]=3.2 [m3p7]=3.7 [m4p2]=4.2 [m4p7]=4.7 )
declare -a mord=( m4p7 m4p2 m3p7 m3p2 m2p7 m2p2 m1p7 m1p2 m0p9 )

# source ${maindir}/bash_scripts/write_inlists.sh # now sourced in do_mesa_run.sh
source ${maindir}/bash_scripts/do_mesa_run.sh
for cb in 0 1 2 3 4 5 6; do
    for mass in "${mord[@]}"; do
        do_mesa_run "${maindir}" "${RUNS}/c${cb}/${mass}" "${mvals[${mass}]}" "${cb}" 0 "master" 0
    done
done
