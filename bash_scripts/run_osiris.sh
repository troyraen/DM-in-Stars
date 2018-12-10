#!/bin/bash

### copies $1/inlist_options_tmplt to $2/inlist_options
### then changes options according to $3, $4, and $5
function write_options_inlist () {
    maindir=$1 # main mesa_wimps dir (e.g. mesaruns)
    RUN=$2 # specific directory for log files
    mass=$3 # floating point number
    cboost=$4 # = integer 0..6
    pgstar=${5:-0} # = 1 generates a movie, default 0

    fopts="${RUN}/inlist_options"
    cp ${maindir}/inlist_options_tmplt ${fopts} # copy template

    # MASS
    sed -i 's/_MASS_/'${mass}'/g' ${fopts}
    mint=$(echo $mass*10 | bc ); mint=$( printf "%.0f" $mint ) # int(mass*10)
    declare -A ma=( [8]=30.E9 [9]=23.E9 )
    if [[ ${ma[$mint]} ]]; then # if mass in ma use value higher than default
        sed -i 's/_MAXAGE_/'${ma[$mint]}'/g' ${fopts}
    else # use default
        sed -i 's/_MAXAGE_/13.E9/g' ${fopts}
    fi

    # CBOOST
    if (( ${cboost} != 0 )); then
        sed -i 's/X_CTRL(1) = 0.E0/X_CTRL(1) = 1.E'${cboost}'/g' ${fopts}
    fi

    # PGSTAR options
    if [ "${pgstar}" = 1 ]; then # save png files
        sed -i 's/pgstar_flag = .false./pgstar_flag = .true./g' ${fopts} # save png files
        cp ${maindir}/inlist_pgstar_my ${RUN}/. # cp custom pgstar inlist
        sed -i 's/read_extra_pgstar_inlist1 = .false./read_extra_pgstar_inlist1 = .true./g' ${fopts} # read it
    fi
}



function rnmesa () {
    maindir=$1 # main mesa_wimps dir (e.g. mesaruns)
    RUN=${maindir}/$2 # specific directory for log files (e.g. RUNS_/SD/C0/m1p4)
    mass=$3 # floating point number
    cboost=$4 # = integer 0..6
    pgstar=${5:-0} # = 1 generates a movie, default 0
    # inlistm=$3 # call this from inlist


    mkdir -p ${RUN}/LOGS ${RUN}/png ${RUN}/photos
    stdout=${RUN}/LOGS/STD.out
    # cp $maindir/$RUNS/$xphoto $maindir/photos/.
    write_options_inlist ${maindir} ${RUN} ${mass} ${cboost} ${pgstar}

    cd ${RUN}
    ${maindir}/star &>> ${stdout}
    # $maindir/re $xphoto &>> LOGS/STD.out
    ${maindir}/bash_scripts/del_dup_mods.sh ${RUN}/LOGS &>> ${stdout}
    # ${maindir}/bash_scripts/data_reduc.sh ${RUN}/LOGS &>> ${stdout}

    #** pgstar movies
    if [ "${pgstar}" = 1 ]; then
        sed -i "/\.png\/png/d" ${stdout} # strip png lines from stdout
        images_to_movie.sh "png/grid1*.png" /LOGS/grid1.mp4 # make movies
        images_to_movie.sh "png/grid2*.png" /LOGS/grid2.mp4
        if [ -f /LOGS/grid1.mp4 ]; then # del png files
            rm -r $maindir/png
        else
            echo "\nSomething went wrong making pgstar movies!\n" &>> ${stdout}
        fi
    fi

    cd ${maindir}
}

### MAIN PROGRAM
### runs mesa models with specified params
maindir="/home/tjr63/mesaruns"
RUNS="RUNS_TEST"

cd ${maindir}
./clean
./mk

rnmesa "${maindir}" "${RUNS}/m1p4" 5.0 1 1


# declare -A ivals=( [enom_true]=.TRUE. [enom_false]=.FALSE.)
# declare -a iord=( inlist_m1p2 inlist_m1p3 inlist_m1p4 inlist_m1p5 )
#
# for inlst in "${iord[@]}"; do
#
#     rnmesa "${maindir}" "${RUNS}/${inlst: -4}" "${inlst}"
#     # echo -e "\n${RUNS}/${inlst: -4}\n"
#
# done
