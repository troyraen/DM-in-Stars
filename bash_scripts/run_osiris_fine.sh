#!/bin/bash

####
# this script does a run for every __iiter__ (see below) element in a
# reversed mord array, starting with index i
    # where i is an argument passed to this script
####

### runs mesa models with specified params
maindir="/home/tjr63/DMS/mesaruns"
cd ${maindir}
RUNS="RUNS_defDM"
i=${1} # where in reversed(mord array) to start (1 => start with last element)

./clean
./mk

source ${maindir}/bash_scripts/do_mesa_run.sh


declare -A mvals=( [m0p83]=0.83 [m0p88]=0.88 [m0p93]=0.93 [m0p98]=0.98 [m1p03]=1.03 [m1p08]=1.08 [m1p13]=1.13 [m1p18]=1.18 [m1p23]=1.23 [m1p28]=1.28 [m1p33]=1.33 [m1p38]=1.38 [m1p43]=1.43 [m1p48]=1.48 [m1p53]=1.53 [m1p58]=1.58 [m1p63]=1.63 [m1p68]=1.68 [m1p73]=1.73 [m1p78]=1.78 [m1p83]=1.83 [m1p88]=1.88 [m1p93]=1.93 [m1p98]=1.98 [m2p03]=2.03 [m2p08]=2.08 [m2p13]=2.13 [m2p18]=2.18 [m2p23]=2.23 [m2p28]=2.28 [m2p33]=2.33 [m2p38]=2.38 [m2p43]=2.43 [m2p48]=2.48 [m2p53]=2.53 [m2p58]=2.58 [m2p63]=2.63 [m2p68]=2.68 [m2p73]=2.73 [m2p78]=2.78 [m2p83]=2.83 [m2p88]=2.88 [m2p93]=2.93 [m2p98]=2.98 [m3p03]=3.03 [m3p08]=3.08 [m3p13]=3.13 [m3p18]=3.18 [m3p23]=3.23 [m3p28]=3.28 [m3p33]=3.33 [m3p38]=3.38 [m3p43]=3.43 [m3p48]=3.48 [m3p53]=3.53 [m3p58]=3.58 [m3p63]=3.63 [m3p68]=3.68 [m3p73]=3.73 [m3p78]=3.78 [m3p83]=3.83 [m3p88]=3.88 [m3p93]=3.93 [m3p98]=3.98 [m4p03]=4.03 [m4p08]=4.08 [m4p13]=4.13 [m4p18]=4.18 [m4p23]=4.23 [m4p28]=4.28 [m4p33]=4.33 [m4p38]=4.38 [m4p43]=4.43 [m4p48]=4.48 [m4p53]=4.53 [m4p58]=4.58 [m4p63]=4.63 [m4p68]=4.68 [m4p73]=4.73 [m4p78]=4.78 [m4p83]=4.83 [m4p88]=4.88 [m4p93]=4.93 [m4p98]=4.98
 )
declare -a mord=( m0p83 m0p88 m0p93 m0p98 m1p03 m1p08 m1p13 m1p18 m1p23 m1p28 m1p33 m1p38 m1p43 m1p48 m1p53 m1p58 m1p63 m1p68 m1p73 m1p78 m1p83 m1p88 m1p93 m1p98 m2p03 m2p08 m2p13 m2p18 m2p23 m2p28 m2p33 m2p38 m2p43 m2p48 m2p53 m2p58 m2p63 m2p68 m2p73 m2p78 m2p83 m2p88 m2p93 m2p98 m3p03 m3p08 m3p13 m3p18 m3p23 m3p28 m3p33 m3p38 m3p43 m3p48 m3p53 m3p58 m3p63 m3p68 m3p73 m3p78 m3p83 m3p88 m3p93 m3p98 m4p03 m4p08 m4p13 m4p18 m4p23 m4p28 m4p33 m4p38 m4p43 m4p48 m4p53 m4p58 m4p63 m4p68 m4p73 m4p78 m4p83 m4p88 m4p93 m4p98 )


# do the runs and skip any where `history.data` exists
iiter=1 # do every iiter^th mass in mord
for cb in $(seq 0 0); do
    for (( idx=${#mord[@]}-${i} ; idx>=0 ; idx=idx-${iiter} )) ; do
        mass="${mord[idx]}"
        do_mesa_run "${maindir}" "${RUNS}/c${cb}/${mass}" "${mvals[${mass}]}" "${cb}" 0 "master" 0 1
    done
done
