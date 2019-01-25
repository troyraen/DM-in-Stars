#!/bin/bash

# This function spawns nscreens screen processes
#   each running a batch of MESA runs using bulk_mesa_run.sh
#
# Function arguments:
#       run_directory and start, increment, stop
#           for complete mass and cboost range sequences
#       nscreens (optional)
#       (maindir and scripts_dir are hardcoded below)
#   and completes a series of mesa runs by calling do_mesa_run.sh.
#   Assumes masses have precision no greater than 2 decimal places

#


RUNS=$1
# for mass seq (complete mass set to run)
mstart=$2
mstop=$3
minc=$4
# mstart=5.0
# mstop=0.8
# minc=-0.05
# for cboost seq
cstart=$5
cinc=$6
cstop=$7
# cstart=3
# cinc=1
# cstop=6
nscreens=${8:-5} # number of screen processes to spawn



#  PREP, separate for screens:
minc_scrn=$(echo ${nscreens} \* ${minc} | bc -l) # minc per screen
# echo $(seq ${mstart} ${minc} ${mstop})
mstart_list=($(seq ${mstart} ${minc} ${mstop})) # separate masses
mstart_list=(${mstart_list[@]:0:${nscreens}})

# Ask user, run make/clean?
maindir="/home/tjr63/mesaruns"
echo
echo "Run clean and make in dir ${maindir}?"
echo "  0 = no; 1 = yes"
read mkcln
if [ "${mkcln}" = 1 ]; then
    cd ${maindir}
    ./clean
    ./mk
fi

scripts_dir="/home/tjr63/mesaruns/bash_scripts"
source ${scripts_dir}/bulk_mesa_run.sh
cd ${scripts_dir}
for ms in "${mstart_list[@]}"; do
    echo "doing bmr_caller ${ms}" >> ${fout}
    screen -dm bulk_mesa_run ${RUNS} ${ms} ${minc_scrn} ${mstop} ${cstart} ${cinc} ${cstop}
	# echo "list $ms"
    # echo $(seq $ms $minc_scrn $mstop)
	# echo
done
