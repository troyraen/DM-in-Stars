#!/bin/bash

####
# This script does a run for every __3rd__ element in a reversed mord array,
# starting with index i, where i is an argument passed to this script.
#
# Example call that does the runs in a background thread and writes the output
# to a file:
# 
# i=1
# nohup nice ./bash_scripts/run_osiris.sh $i &>> run_osiris.out &
#
# To run all masses in mord, call this script three times, with
# i = 1, 2, and 3.
#
# Be sure to look at the arguments to to the do_mesa_run command below.
# They determine many of the run parameters.
#
# We reverse the mord array because the higher mass models run a lot faster,
# so might as well do them first so we can look at them sooner.
####

### runs mesa models with specified params
maindir="/home/tjr63/DMS/mesaruns"
cd ${maindir}
RUNS="RUNS_defDM"
i=${1} # where in reversed(mord array) to start (1 => start with last element)

./clean
./mk

source ${maindir}/bash_scripts/do_mesa_run.sh


declare -A mvals=( [m0p85]=0.85 [m0p90]=0.90 [m0p95]=0.95 [m1p00]=1.00 [m1p05]=1.05 [m1p10]=1.10 [m1p15]=1.15 [m1p20]=1.20 [m1p25]=1.25 [m1p30]=1.30 [m1p35]=1.35 [m1p40]=1.40 [m1p45]=1.45 [m1p50]=1.50 [m1p55]=1.55 [m1p60]=1.60 [m1p65]=1.65 [m1p70]=1.70 [m1p75]=1.75 [m1p80]=1.80 [m1p85]=1.85 [m1p90]=1.90 [m1p95]=1.95 [m2p00]=2.00 [m2p05]=2.05 [m2p10]=2.10 [m2p15]=2.15 [m2p20]=2.20 [m2p25]=2.25 [m2p30]=2.30 [m2p35]=2.35 [m2p40]=2.40 [m2p45]=2.45 [m2p50]=2.50 [m2p55]=2.55 [m2p60]=2.60 [m2p65]=2.65 [m2p70]=2.70 [m2p75]=2.75 [m2p80]=2.80 [m2p85]=2.85 [m2p90]=2.90 [m2p95]=2.95 [m3p00]=3.00 [m3p05]=3.05 [m3p10]=3.10 [m3p15]=3.15 [m3p20]=3.20 [m3p25]=3.25 [m3p30]=3.30 [m3p35]=3.35 [m3p40]=3.40 [m3p45]=3.45 [m3p50]=3.50 [m3p55]=3.55 [m3p60]=3.60 [m3p65]=3.65 [m3p70]=3.70 [m3p75]=3.75 [m3p80]=3.80 [m3p85]=3.85 [m3p90]=3.90 [m3p95]=3.95 [m4p00]=4.00 [m4p05]=4.05 [m4p10]=4.10 [m4p15]=4.15 [m4p20]=4.20 [m4p25]=4.25 [m4p30]=4.30 [m4p35]=4.35 [m4p40]=4.40 [m4p45]=4.45 [m4p50]=4.50 [m4p55]=4.55 [m4p60]=4.60 [m4p65]=4.65 [m4p70]=4.70 [m4p75]=4.75 [m4p80]=4.80 [m4p85]=4.85 [m4p90]=4.90 [m4p95]=4.95 [m5p00]=5.00 )
declare -a mord=( m0p85 m0p90 m0p95 m1p00 m1p05 m1p10 m1p15 m1p20 m1p25 m1p30 m1p35 m1p40 m1p45 m1p50 m1p55 m1p60 m1p65 m1p70 m1p75 m1p80 m1p85 m1p90 m1p95 m2p00 m2p05 m2p10 m2p15 m2p20 m2p25 m2p30 m2p35 m2p40 m2p45 m2p50 m2p55 m2p60 m2p65 m2p70 m2p75 m2p80 m2p85 m2p90 m2p95 m3p00 m3p05 m3p10 m3p15 m3p20 m3p25 m3p30 m3p35 m3p40 m3p45 m3p50 m3p55 m3p60 m3p65 m3p70 m3p75 m3p80 m3p85 m3p90 m3p95 m4p00 m4p05 m4p10 m4p15 m4p20 m4p25 m4p30 m4p35 m4p40 m4p45 m4p50 m4p55 m4p60 m4p65 m4p70 m4p75 m4p80 m4p85 m4p90 m4p95 m5p00)


# do the runs and skip any where `history.data` exists
for cb in $(seq 0 6); do
    for (( idx=${#mord[@]}-${i} ; idx>=0 ; idx=idx-3 )) ; do
        mass="${mord[idx]}"
        do_mesa_run "${maindir}" "${RUNS}/c${cb}/${mass}" "${mvals[${mass}]}" "${cb}" 0 "master" 0 1
    done
done
