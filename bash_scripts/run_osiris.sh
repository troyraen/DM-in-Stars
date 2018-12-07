#!/bin/bash

function rnmesa () {
    maindir=$1 # main mesa_wimps dir (e.g. mesaruns)
    RUN=${maindir}/$2 # specific directory for log files
    inlistm=$3 # call this from inlist

    mkdir -p ${RUN}/LOGS ${RUN}/png ${RUN}/photos
    stdout=${RUN}/LOGS/STD.out
    # cp $maindir/$RUNS/$xphoto $maindir/photos/.

    # CHANGE INLIST
    cp ${maindir}/inlist_tmplt ${maindir}/inlist
    sed -i 's/inlist_1/'${inlistm}'/g' $maindir/inlist
    cp ${maindir}/inlist ${maindir}/${inlistm} ${RUN}/LOGS/.

    cd ${RUN}
    ${maindir}/rn &>> ${stdout}
    # $maindir/re $xphoto &>> LOGS/STD.out
    ${maindir}/bash_scripts/del_dup_mods.sh ${RUN}/LOGS &>> ${stdout}
    # ${maindir}/bash_scripts/data_reduc.sh ${RUN}/LOGS &>> ${stdout}

    #** pgstar movies
    # $maindir/pgstar_movie grid1
    # mv $maindir/movie.mp4 $maindir/LOGS/grid1.mp4
    # $maindir/pgstar_movie grid2
    # mv $maindir/movie.mp4 $maindir/LOGS/grid2.mp4
    # rm -r $maindir/png
    cd ${maindir}
}

# declare -A ivals=( [enom_true]=.TRUE. [enom_false]=.FALSE.)
declare -a iord=( inlist_m1p2 inlist_m1p3 inlist_m1p4 inlist_m1p5 )

maindir="/home/tjr63/mesaruns"
RUNS="RUNS_convCore"

for inlst in "${iord[@]}"; do

    rnmesa "${maindir}" "${RUNS}/${inlst: -4}" "${inlst}"
    # echo -e "\n${RUNS}/${inlst: -4}\n"

done
