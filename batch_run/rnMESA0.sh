##!/usr/local/bin/bash
#!/bin/bash

function check_okay {
	if [ $? -ne 0 ]
	then
		exit 1
	fi
}


export MESA_DIR=/home/tjr63/mesa-r10398
export OMP_NUM_THREADS=1
export MESA_BASE=/home/tjr63/mesaruns
# !!! If you change MESA_BASE you must change the file paths in inlist and condor_wrapper !!!
export MESA_INLIST=$MESA_BASE/inlist
export MESA_RUN=$MESA_BASE/RUNS_xETx
#export MESA_RUN=/home/tjr63/sand
rnmesa=rnMESA0
# logfile=$MESA_BASE/batch_run/logs/$rnmesa.out
cp $MESA_BASE/batch_run/$rnmesa.sh $MESA_BASE/batch_run/logs/.

declare -A svals=( [SD]=.TRUE. [SI]=.FALSE. )
declare -a sord=( SD )
declare -A cbvals=( [c0]=0.D0 [c1]=1.D1 [c2]=1.D2 [c3]=1.D3 [c4]=1.D4 [c5]=1.D5 [c6]=1.D6 )
#declare -a cord=( c0 c1 c2 c3 c4 c5 c6 )
declare -a cord=( c5 )
declare -A mvals=( [m0p8]=0.8D0 [m5p0]=5.0D0 [m3p0]=3.0D0 [m4p3]=4.3D0 [m4p4]=4.4D0 [m4p5]=4.5D0 [m4p6]=4.6D0 [m4p7]=4.7D0 [m4p8]=4.8D0 [m4p9]=4.9D0 )
declare -a mord=( m0p8 )
declare -A ovals=( [na]=neither [n]=net [d]=diffus [b]=both )
declare -a oord=( na )
declare -A rsevals=( [xEOG]=xEOG [xETx]=xETxtol )
declare -a rseord=( xEOG xETx )

for xE in "${rseord[@]}"; do
for oth in "${oord[@]}"; do
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

			Ldir=$MESA_RUN/$mass/${cdir}_$xE
            mkdir -pm 777 $Ldir
                cp $MESA_BASE/batch_run/xinlist_template $Ldir/inlist_cluster
				mist_inlist=inlist_mymist_${ovals[$oth]}
				cp $MESA_BASE/$mist_inlist $Ldir/$mist_inlist
				cp $MESA_BASE/src/run_star_extras_${rsevals[$xE]}.f $MESA_BASE/src/run_star_extras.f
                check_okay

				$MESA_BASE/make/make clean
				$MESA_BASE/make/make
				check_okay

                cd $Ldir

                sed -i 's/imass_/'${mvals[$mass]}'/g; s/maxage_/'$ma'/g; s/oenergy_/'$oe'/g; s/cboost_/'${cbvals[$cdir]}'/g; s/SD_/'${svals[$spin]}'/g; s/_OTHER/_'${ovals[$oth]}'/g' inlist_cluster
                check_okay

                $MESA_BASE/star &> STD.out
                check_okay
				$MESA_BASE/bash_scripts/del_dup_mods.sh $Ldir &>> STD.out # &>> $logfile # delete duplicate models
				check_okay

                cd $MESA_RUN
        done
    done
done
done
done
