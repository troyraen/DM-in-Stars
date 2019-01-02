#!/bin/bash

maindir="/home/tjr63/mesaruns"
RUNS="RUNS_pgstarGrid2"
source ${maindir}/bash_scripts/write_inlists


###... MOVED TO EXTERNAL FILE ...###
# ### copies $1/inlist_options_tmplt to $2/inlist_options
# ### then changes options according to $3, $4, and $5
# function write_inlists () {
#     maindir=$1 # main mesa_wimps dir (e.g. mesaruns)
#     RUN=$2 # specific directory for log files
#     mass=$3 # floating point number
#     cboost=$4 # = integer 0..6
#     pgstar=${5:-0} # = 1 generates a movie, default 0
#     inlist_master=${6:-"master"} # inlist_$6 will be used as base inlist
#     stop_leaveMS=${7:-0} # = 1 stops the run when h1 frac < 1.e-3
#
#     # INLISTS
#     fopts="${RUN}/inlist_options"
#     cp ${maindir}/inlist_options_tmplt ${fopts} # copy template
#     cp ${maindir}/inlist_${inlist_master} ${RUN}/inlist # copy main inlist
#
#     ### Change inlist_options properties:
#
#     # MASS
#     sed -i 's/_MASS_/'${mass}'/g' ${fopts}
#     mint=$(echo $mass*10 | bc ); mint=$( printf "%.0f" $mint ) # int(mass*10)
#     declare -A ma=( [8]=30.E9 [9]=23.E9 )
#     if [[ ${ma[$mint]} ]]; then # if mass in ma use value higher than default
#         sed -i 's/_MAXAGE_/'${ma[$mint]}'/g' ${fopts}
#     else # use default
#         sed -i 's/_MAXAGE_/13.E9/g' ${fopts}
#     fi
#
#     # CBOOST
#     if (( ${cboost} != 0 )); then
#         sed -i 's/use_other_energy_implicit = .false./use_other_energy_implicit = .true./g' ${fopts}
#         sed -i 's/X_CTRL(1) = 0.E0/X_CTRL(1) = 1.E'${cboost}'/g' ${fopts}
#     fi
#
#     # PGSTAR options
#     if [ "${pgstar}" = 1 ]; then # save png files
#         sed -i 's/pgstar_flag = .false./pgstar_flag = .true./g' ${fopts} # save png files
#         cp ${maindir}/inlist_pgstar_my ${RUN}/. # cp custom pgstar inlist
#         sed -i 's/read_extra_pgstar_inlist2 = .false./read_extra_pgstar_inlist2 = .true./g' ${fopts} # read it
#     fi
#
#     # stop after leave MS
#     if [ "${stop_leaveMS}" = 1 ]; then
#         sed -i 's/! xa_central_lower_limit/xa_central_lower_limit/g' ${fopts} # uncomment these 2 lines
#     fi
#
#     echo "Wrote inlists (${fopts})"
# }



function rnmesa () {
    maindir=$1 # main mesa_wimps dir (e.g. mesaruns)
    RUN=${maindir}/$2 # specific directory for log files (e.g. RUNS_/SD/C0/m1p4)
    mass=$3 # floating point number
    cboost=$4 # = integer 0..6
    pgstar=${5:-0} # = 1 generates a movie, default 0
    inlist_master=${6:-"master"} # inlist_$6 will be used as base inlist
    stop_leaveMS=${7:-0} # = 1 stops the run when h1 frac < 1.e-3

    ### PREP
    echo
    if [ -d "${RUN}" ]; then
        if [ -f "${RUN}/LOGS/history.data" ]; then
            tm=$(date +"%m%d%y_%H%M")
            mv ${RUN} ${RUN}_ow_${tm}  # move anything that's already here
        else
            rm -r ${RUN} # no previous history.data file exists. delete the folder
        fi
    fi
    mkdir -p ${RUN}/LOGS ${RUN}/png ${RUN}/photos
    stdout=${RUN}/LOGS/STD.out
    # cp $maindir/$RUNS/$xphoto $maindir/photos/.
    write_inlists ${maindir} ${RUN} ${mass} ${cboost} ${pgstar} ${inlist_master} ${stop_leaveMS}

    ### RUN
    echo "Running MESA ..."
    cd ${RUN}
    ${maindir}/star &>> ${stdout}
    # $maindir/re $xphoto &>> LOGS/STD.out
    ${maindir}/bash_scripts/del_dup_mods.sh ${RUN}/LOGS &>> ${stdout}
    # ${maindir}/bash_scripts/data_reduc.sh ${RUN}/LOGS &>> ${stdout}

    ### Finish pgstar movies
    if [ "${pgstar}" = 1 ]; then
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



### MAIN PROGRAM
### runs mesa models with specified params
mkcln="${1:-1}" #  = 1 will execute ./mk and ./clean

cd ${maindir}
if [ "${mkcln}" = 1 ]; then
        ./clean
        ./mk
fi

# declare -A mvals=( [m0p8]=0.8 [m1p0]=1.0 [m1p1]=1.1 [m1p2]=1.2 [m1p3]=1.3 [m1p4]=1.4 [m1p6]=1.6 [m2p0]=2.0 [m3p0]=3.0 [m4p0]=4.0 )
# declare -a mord=( m0p8 m1p0 m1p1 m1p2 m1p3 m1p4 m1p6 m2p0 m3p0 m4p0 )
declare -A mvals=( [m0p8]=0.8 [m1p0]=1.0 [m1p3]=1.3 [m2p5]=2.5 [m3p5]=3.5 [m4p5]=4.5 )
declare -a mord=( m4p5 m3p5 m2p5 m1p3 m1p0 m0p8 )

for cb in 0 3 6; do
    for mass in "${mord[@]}"; do
        rnmesa "${maindir}" "${RUNS}/c${cb}/${mass}" "${mvals[${mass}]}" "${cb}" 1 "master" 0
    done
done
