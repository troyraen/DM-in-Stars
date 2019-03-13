#!/bin/bash

# Uses inlist_master_profiles to do a MESA run.
# Saves a model ever step, with max numprofs
# Stops at model_number = stopmod.

cb=$1 # cboost as integer
mass=$2 # mass for dir (e.g. m3p50)
mval=$3 # mass as float (e.g. 3.5)
stopmod=$4 # model number to stop the run
numprofs=$5 # max number of profiles to save


maindir="/home/tjr63/mesaruns"
cd ${maindir}
RUNS="RUNS_2test_final/profile_runs"
specRUNS="${RUNS}/c${cb}/${mass}"

# Ask user, run make/clean?
echo
echo "Run clean and make in dir ${maindir}?"
echo "  0 = no; 1 = yes"
read mkcln
if [ "${mkcln}" = 1 ]; then
        ./clean
        ./mk
fi

# Set max number of profiles to save
finlist="inlist_master_profiles"
sed -i 's/__NUM_PROFILES__/'${numprofs}'/g' ${finlist}

source ${maindir}/bash_scripts/do_mesa_run.sh
do_mesa_run "${maindir}" "${specRUNS}" "${mval}" "${cb}" 1 "master_profiles" "${stopmod}"

# Invert: Set max number of profiles to save
sed -i 's/'${numprofs}'/__NUM_PROFILES__/g' ${finlist}

# rename directory
mv "${maindir}/${specRUNS}" "${maindir}/${specRUNS}_stopmod${stopmod}"
