##!/usr/local/bin/bash
#!/bin/bash

function check_okay {
	if [ $? -ne 0 ]
	then
		exit 1
	fi
}


export MESA_DIR=/home/tjr63/mesa-r9793
export OMP_NUM_THREADS=1
export MESA_BASE=/home/tjr63/implicit_test2i
# !!! If you change MESA_BASE you must change the file paths in inlist and condor_wrapper !!!
export MESA_INLIST=$MESA_BASE/inlist
export MESA_RUN=$MESA_BASE/RUNS
#export MESA_RUN=/home/tjr63/sand

declare -A svals=( [SD]=.TRUE. [SI]=.FALSE. )
declare -a sord=( SD )
declare -A cbvals=( [c0]=0.D0 [c1]=1.D1 [c2]=1.D2 [c3]=1.D3 [c4]=1.D4 [c5]=1.D5 [c6]=1.D6 )
declare -A mvals=( [m7p3]=7.3D0 [m6p3]=6.3D0 [m5p3]=5.3D0 [m4p3]=4.3D0 [m3p3]=3.3D0 [m2p3]=2.3D0 [m1p3]=1.3D0 )

### finish c0's that didn't run before
declare -a cord=( c5 )
declare -a mord=( m3p3 m2p3 m1p3 )

for spin in "${sord[@]}"; do
    for cdir in "${cord[@]}"; do
		oe=$([ $cdir = c0 ] && echo ".false." || echo ".true.") # use_other_energy_implicit=.false. if c0 else .true.
        for mass in "${mord[@]}"; do
			if [ $mass = m0p8 ]; then # set max_age depending on mass
				ma=22.D9
			elif [ $mass = m0p9 ]; then
				ma=15.D9
			else
				ma=13.D9
			fi

            mkdir -pm 777 $MESA_RUN/$spin/$cdir/$mass
                cp $MESA_BASE/implicit_test_files/xinlist_template $MESA_RUN/$spin/$cdir/$mass/inlist_cluster
                check_okay
                cd $MESA_RUN/$spin/$cdir/$mass

                sed -i 's/imass_/'${mvals[$mass]}'/g; s/maxage_/'$ma'/g; s/oenergy_/'$oe'/g; s/cboost_/'${cbvals[$cdir]}'/g; s/SD_/'${svals[$spin]}'/g' inlist_cluster
                check_okay

                $MESA_BASE/star
                check_okay

                cd $MESA_RUN
        done
    done
done

### finish the rest of the cboosts
declare -a cord=( c6 )
declare -a mord=( m4p3 m3p3 m2p3 m1p3 )

for spin in "${sord[@]}"; do
    for cdir in "${cord[@]}"; do
		oe=$([ $cdir = c0 ] && echo ".false." || echo ".true.") # use_other_energy_implicit=.false. if c0 else .true.
        for mass in "${mord[@]}"; do
			if [ $mass = m0p8 ]; then # set max_age depending on mass
				ma=22.D9
			elif [ $mass = m0p9 ]; then
				ma=15.D9
			else
				ma=13.D9
			fi

            mkdir -pm 777 $MESA_RUN/$spin/$cdir/$mass
                cp $MESA_BASE/implicit_test_files/xinlist_template $MESA_RUN/$spin/$cdir/$mass/inlist_cluster
                check_okay
                cd $MESA_RUN/$spin/$cdir/$mass

                sed -i 's/imass_/'${mvals[$mass]}'/g; s/maxage_/'$ma'/g; s/oenergy_/'$oe'/g; s/cboost_/'${cbvals[$cdir]}'/g; s/SD_/'${svals[$spin]}'/g' inlist_cluster
                check_okay

                $MESA_BASE/star
                check_okay

                cd $MESA_RUN
        done
    done
done
