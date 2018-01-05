##!/usr/local/bin/bash
#!/bin/bash


export MESA_DIR=/home/tjr63/mesa-r9793
export OMP_NUM_THREADS=1
export MESA_BASE=/home/tjr63/mesa_wimps
export MESA_INLIST=$MESA_BASE/inlist
#export MESA_RUN=/home/tjr63/isoc
export MESA_RUN=/home/tjr63/sand

declare -A svals=( [SD]=.TRUE. [SI]=.FALSE. )
declare -a sord=(SD SI)
declare -A cbvals=( [c0]=0.D0 [c1]=1.D1 [c2]=1.D2 [c3]=1.D3 [c4]=1.D4 [c5]=1.D5 )
declare -a cord=( c0 c1 c2 c3 )
# run the rest of cbvals later
declare -A mvals=( [m4p8]=4.8D0 [m3p8]=3.8D0 [m2p8]=2.8D0 [m1p8]=1.8D0 )
declare -a mord=( m4p8 m3p8 m2p8 m1p8 )


for spin in "${sord[@]}"; do
    for cdir in "${cord[@]}"; do
        for mass in "${mord[@]}"; do
            mkdir -pm 777 $MESA_RUN/$spin/$cdir/$mass
                cp $MESA_RUN/xinlist_template $MESA_RUN/$spin/$cdir/$mass/inlist_cluster
                cd $MESA_RUN/$spin/$cdir/$mass
                    sed -i 's/s_c_m_/'$spin$cdir$mass'/g; s/cboost_/'${cbvals[$cdir]}'/g; s/imass_/'${mvals[$mass]}'/g; s/SD_/'${svals[$spin]}'/g' inlist_cluster

                   $MESA_BASE/star

                cd $MESA_RUN
        done
    done
done
