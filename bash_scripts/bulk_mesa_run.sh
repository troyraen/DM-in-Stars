#!/bin/bash

#*** ./clean and ./mk should be done PRIOR TO RUNNING THIS SCRIPT ***#
#
# This script takes arguments
#       run_directory and start, increment, stop for mass and cboost range sequences
#       (maindir is hardcoded below)
#   and completes a series of mesa runs by calling do_mesa_run.sh.
#   Assumes masses have precision no greater than 2 decimal places
#
# Call using the following syntax to run this script in a detached screen:
#   screen -dm ./basepath/bulk_mesa_run.sh arguments
#

# function bulk_mesa_run () {
# --- Arguments
# set directories
maindir="/home/tjr63/mesaruns"
RUNS=$1
# other arguments
mstart=$2
minc=$3
mstop=$4
cstart=$5
cinc=$6
cstop=$7
# ---


#--- Create arrays for masses and cboost
declare -A mvals=()
declare -a mord=()

for fmass in $(seq -f "%1.2f" ${mstart} ${minc} ${mstop}); do
	mr="${fmass:0:1}"
	mp="${fmass:2}"
	smass="m${mr}p${mp}"

	mord=("${mord[@]}" "${smass}")
	mvals["${smass}"]="${fmass}"
done

# for mass in "${mord[@]}"; do
# 	echo "${mass}" "${mvals[${mass}]}"
# done

cvals=($(seq ${cstart} ${cinc} ${cstop}))
# ---


#--- Do MESA run_star_extras
source ${maindir}/bash_scripts/do_mesa_run.sh
pgstar=0
inlist="master"
stop_TAMS=0
for cb in "${cvals[@]}"; do
    for mass in "${mord[@]}"; do
        do_mesa_run "${maindir}" "${RUNS}/c${cb}/${mass}" "${mvals[${mass}]}" "${cb}" "${pgstar}" "${inlist}" "${stop_TAMS}"
            # mkdir -p ${maindir}/${RUNS}/c${cb}/${mass}
    done
done

# }
