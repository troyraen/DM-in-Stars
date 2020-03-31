#!/bin/bash

### copies $6 master inlist to $2/inlist
### copies $1/inlist_options_tmplt to $2/inlist_options
### then changes options according to $3, $4, $7 ...
function write_inlists () {
    if [ $# -eq 0 ]
      then
        echo "*********** Must supply write_inlists.sh with directories and parameter arguments. ***********"
        exit 1
    fi

    maindir=$1 # main dir for runs (e.g. mesaruns)
    RUN=$2 # specific directory for log files
    mass=$3 # floating point number
    cboost=$4 # = integer 0..6
    pgstar=${5:-0} # = 1 generates a movie, default 0
    inlist_master=${6:-"master"} # inlist_$6 will be used as base inlist
    stop_TAMS=${7:-0} # = 0 does nothing
                      # = 1 stops the run when h1 frac < 1.e-12
                      # > 1 stops the run when model_number == $7

    # INLISTS
    fopts="${RUN}/inlist_options"
    cp ${maindir}/inlist_options_tmplt ${fopts} # copy template
    cp ${maindir}/inlist_${inlist_master} ${RUN}/inlist # copy main inlist

    ### Change inlist_options properties:

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
        sed -i 's/use_other_energy_implicit = .false./use_other_energy_implicit = .true./g' ${fopts}
        sed -i 's/X_CTRL(1) = 0.E0/X_CTRL(1) = 1.E'${cboost}'/g' ${fopts}
    fi

    # PGSTAR options
    if [ "${pgstar}" = 1 ]; then # save png files
        sed -i 's/pgstar_flag = .false./pgstar_flag = .true./g' ${fopts} # save png files
        cp ${maindir}/inlist_pgstar_my ${RUN}/. # cp custom pgstar inlist
        sed -i 's/read_extra_pgstar_inlist2 = .false./read_extra_pgstar_inlist2 = .true./g' ${fopts} # read it
    fi

    # stop condition
    if [ "${stop_TAMS}" = 0 ]; then
        : # pass
    elif [ "${stop_TAMS}" = 1 ]; then # stop at TAMS
        sed -i 's/! xa_central_lower_limit/xa_central_lower_limit/g' ${fopts} # uncomment these 2 lines
    else # stop at model_number = stop_TAMS
        sed -i 's/max_model_number = -1/max_model_number = '${stop_TAMS}'/g' ${fopts} # set max model number
    fi

    echo "Wrote inlists (${fopts})"
}
