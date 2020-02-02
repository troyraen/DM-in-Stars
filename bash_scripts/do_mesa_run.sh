#!/bin/bash


function do_mesa_run () {
    maindir=$1 # main mesa_wimps dir (e.g. mesaruns)
    RUN=${maindir}/$2 # specific directory for log files (e.g. RUNS_/SD/C0/m1p4)
    mass=$3 # floating point number
    cboost=$4 # = integer 0..6
    pgstar=${5:-0} # = 1 generates a movie, default 0
    inlist_master=${6:-"master"} # inlist_$6 will be used as base inlist
    stop_TAMS=${7:-0} # = 1 stops the run when h1 frac < 1.e-12
    skip_existing=${8:-0} # = 1 exits the run if ${RUN}/LOGS/history.data exists

    ### PREP
    source ${maindir}/bash_scripts/write_inlists.sh

    echo
    if [ -d "${RUN}" ]; then # check for existing history file in $RUN
        if [ -f "${RUN}/LOGS/history.data" ]; then
            if [ "${skip_existing}" = 1 ]; then # exit the run
                echo "Run exists, skipping: ${mass}M_sun cb${cboost}"
                echo
                return 0

            else # move what's already here
                tm=$(date +"%m%d%y_%H%M")
                mv ${RUN} ${RUN}_ow_${tm}
            fi
        else
            rm -r ${RUN} # no previous history.data file exists. delete the folder
        fi
    fi
    mkdir -p ${RUN}/LOGS ${RUN}/png ${RUN}/photos
    stdout=${RUN}/LOGS/STD.out
    # cp $maindir/$RUNS/$xphoto $maindir/photos/.
    write_inlists ${maindir} ${RUN} ${mass} ${cboost} ${pgstar} ${inlist_master} ${stop_TAMS}

    ### RUN
    echo "Running MESA in dir ${RUN}"
    echo "  ... ... ..."
    cd ${RUN}
    ${maindir}/star &>> ${stdout}
    # $maindir/re $xphoto &>> LOGS/STD.out
    ${maindir}/bash_scripts/del_dup_mods.sh ${RUN}/LOGS &>> ${stdout}
    # ${maindir}/bash_scripts/data_reduc.sh ${RUN}/LOGS &>> ${stdout}

    ### Finish pgstar movies
    if [ "${pgstar}" = 1 ]; then
        echo "Making pgstar movies"
        images_to_movie.sh "png/grid1*.png" ./LOGS/grid1_${mass}c${cb}.mp4 # make movies
        images_to_movie.sh "png/grid2*.png" ./LOGS/grid2_${mass}c${cb}.mp4
        if [ -f ./LOGS/grid2_${mass}c${cb}.mp4 ]; then
            echo "Pgstar movies created." &>> ${stdout}
            sed -i "/\.png\/png/d" ${stdout} # strip png lines from stdout
            rm -r ${RUN}/png # del png files
        else
            echo "Something went wrong making pgstar movies!" &>> ${stdout}
        fi
    fi

    ### CLEANUP
    cd ${maindir}
    echo "Finished ${mass}M_sun cb${cboost}"
    echo
}
