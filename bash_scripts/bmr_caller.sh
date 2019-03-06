#!/bin/bash

# This function spawns nscreens screen processes
#   each running a batch of MESA runs using bulk_mesa_run.sh.
#   Function is called in last line of file
#
# Function arguments:
#       run_directory and start, increment, stop
#           for complete mass and cboost range sequences
#       nscreens (optional, default=5)
#       (maindir and scripts_dir are hardcoded below)
#   and completes a series of mesa runs by calling do_mesa_run.sh.
#   Assumes masses have precision no greater than 2 decimal places
#
# Example Usage:
#   ./bash_scripts/bmr_caller.sh RUNS_2test_final 0.8 0.05 5.0 0 1 6 4

export OMP_NUM_THREADS=6

function bmr_caller () {
    RUNS=$1
    # for mass seq (complete mass set to run)
    mstart=$2
    minc=$3
    mstop=$4
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
    # source ${scripts_dir}/bulk_mesa_run.sh
    cd ${scripts_dir}
    for ms in "${mstart_list[@]}"; do
        screenname="_bmr_StartMass_${ms}"
        echo
        echo "bmr_caller calling bulk_mesa_run with start mass: ${ms} Msun."
        echo "  Runs folder is ${RUNS}."
        echo "  Screen name is ${screenname}"
        screen -dm -S ${screenname} ./bulk_mesa_run.sh ${RUNS} ${ms} ${minc_scrn} ${mstop} ${cstart} ${cinc} ${cstop}
        # screen -dm bulk_mesa_run ${RUNS} ${ms} ${minc_scrn} ${mstop} ${cstart} ${cinc} ${cstop}
    	# echo "list $ms"
        # echo $(seq $ms $minc_scrn $mstop)
    	# echo
    done

}


bmr_caller "RUNS_test_final" 5.0 -0.05 0.79 3 1 6
