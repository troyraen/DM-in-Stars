#!/bin/bash

### runs mesa models with specified params
maindir="/home/tjr63/DMS/mesaruns"
cd ${maindir}
RUNS="RUNS_largeLH_mesa-r12115" # using the newest MESA release r12115

# Ask user, run make/clean?
echo
echo "Run clean and make in dir ${maindir}?"
echo "  0 = no; 1 = yes"
read mkcln
if [ "${mkcln}" = 1 ]; then
        ./clean
        ./mk
fi

# declare -A mvals=( [m0p8]=0.8 [m1p0]=1.0 [m1p1]=1.1 [m1p2]=1.2 [m1p3]=1.3 [m1p4]=1.4 [m1p6]=1.6 [m2p0]=2.0 [m3p0]=3.0 [m4p0]=4.0 )
# declare -a mord=( m0p8 m1p0 m1p1 m1p2 m1p3 m1p4 m1p6 m2p0 m3p0 m4p0 )
declare -A mvals=( [m2p0]=2.0 [m3p0]=3.0 [m4p0]=4.0 [m5p0]=5.0 )
declare -a mord=( m5p0 m4p0 m3p0 m2p0 )
# declare -A mvals=( [m0p8]=0.8 [m1p0]=1.0 [m1p3]=1.3 [m2p5]=2.5 [m3p5]=3.5 [m4p5]=4.5 )
# declare -a mord=( m1p0 )

# source ${maindir}/bash_scripts/write_inlists.sh # now sourced in do_mesa_run.sh
source ${maindir}/bash_scripts/do_mesa_run.sh
for cb in 0 4 6; do
    for mass in "${mord[@]}"; do
        do_mesa_run "${maindir}" "${RUNS}/c${cb}/${mass}" "${mvals[${mass}]}" "${cb}" 0 "master" 0
    done
done
