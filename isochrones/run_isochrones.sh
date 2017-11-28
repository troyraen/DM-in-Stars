#!/bin/bash


export MESA_DIR=/home/tjr63/mesa-r9793
export OMP_NUM_THREADS=12
export MESA_BASE=/home/tjr63/mesa_wimps
export MESA_INLIST=$MESA_BASE/inlist
export MESA_RUN=/home/tjr63/mesarun

#$MESA_BASE/.clean
#$MESA_BASE/.mk

declare -A svals=( [SD]=.TRUE. [SI]=.FALSE.)
declare -A cbvals=( [c0]=0.D0 [c1]=1.D1 [c2]=1.D2 )
declare -A mvals=( [m0p5]=0.5D0 [m1p0]=1.D0 [m1p5]=1.5D0 )



for spin in ${!svals[@]}; do
    mkdir $MESA_RUN/${spin}
    for cdir in ${!cbvals[@]}; do
        mkdir $MESA_RUN/${spin}/${cdir}
        for mass in ${!mvals[@]}; do
            mkdir $MESA_RUN/${spin}/${cdir}/${mass}
                cp $MESA_RUN/xinlist_template $MESA_RUN/${spin}/${cdir}/${mass}/inlist_cluster
                cd $MESA_RUN/${spin}/${cdir}/${mass}
                sed -i 's/s_c_m_/'${spin}${cdir}${mass}'/g; s/cboost_/'${cbvals[${cdir}]}'/g; s/imass_/'${mvals[${mass}]}'/g; s/S$

#               $MESA_BASE/star

                cd ..
                cd ..
                cd ..
        done
    done
done
