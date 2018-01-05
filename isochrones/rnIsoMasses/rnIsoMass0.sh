#!/usr/local/bin/bash



export MESA_DIR=/home/tjr63/mesa-r9793
export OMP_NUM_THREADS=1
export MESA_BASE=/home/tjr63/mesa_wimps
export MESA_INLIST=$MESA_BASE/inlist
export MESA_RUN=/home/tjr63/isoc


declare -A svals=( [SD]=.TRUE., [SI]=.FALSE. )
declare -A cbvals0=( [c0]=0.D0, [c1]=1.D1, [c2]=1.D2, [c3]=1.D3 )
declare -A cbvals1=( [c4]=1.D4 )
declare -A cbvals2=( [c5]=1.D5 )
declare -A cblist=( [0]=cbvals0, [1]=cbvals1, [2]=cbvals2 )
declare -A mvals=( 0.9, 0.8 )


for cbvals in ${!cblist[@]}; do
    for spin in ${!svals[@]}; do
        for cdir in ${!cbvals[@]}; do
            for mass in ${!mvals[@]}; do
                mkdir -p $MESA_RUN/${spin}/${cdir}/${mass}
                    cp $MESA_RUN/xinlist_template $MESA_RUN/${spin}/${cdir}/${mass}/inlist_cluster
                    cd $MESA_RUN/${spin}/${cdir}/${mass}
#                    sed -i 's/s_c_m_/'${spin}${cdir}${mass}'/g; s/cboost_/'${cbvals[${cdir}]}'/g; s/imass_/'${mvals[${mass}]}'/g; s/SD_/'${svals[${spin}]}'/g' inlist_cluster
                    sed -i 's cboost_ '${cbvals[${cdir}]}' g; s imass_ '${mvals[${mass}]}' g; s SD_ '${svals[${spin}]}' g' inlist_cluster

#                   $MESA_BASE/star

                    cd ..
                    cd ..
                    cd ..
            done
        done
    done
done
