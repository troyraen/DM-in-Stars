#!/bin/bash

# declare -A ivals=( [mlt]=mlt [sc]=semiconvection [th]=thermohaline [jina]=jina [mts]=mesh_timestep [opa]=opacity [ovs]=overshoot [ow]=other_wind [opos]=opacity_overshoot [oth]=other [OG]=OG)
# declare -A ivals=( [mlt]=mlt [sc]=semiconvection [th]=thermohaline [opa]=opacity [oth]=other [OG]=OG [pre]=preMIST [lim]=limits [hms]=hook_preMST [otm]=op_thrm_mlt)
# declare -a iord=( pre opa mlt sc th lim hms oth)
# declare -a iord=(OG pre otm)

# declare -A ivals=( [base]=base [enlr]=en_lr [enom]=enom [lr]=lr )
# declare -a iord=(base enom)

declare -A ivals=( [enom_true]=.TRUE. [enom_false]=.FALSE.)
declare -a iord=(enom_false enom_true)

maindir=/home/tjr63/mesaruns
outfl=$maindir/LOGS/STD.out
RUNS=RUNS_plot_Tx_energies # RUNS_xLdivLnuc_test
xphoto=x773_xLdivLnuc_GT_0p085_photo

for inlst in "${iord[@]}"; do
    mkdir $maindir/LOGS $maindir/png $maindir/photos
    cp $maindir/$RUNS/$xphoto $maindir/photos/.

    # CHANGE INLIST
    # cp $maindir/inlist_${ivals[$inlst]} $maindir/inlist
    # cp $maindir/inlist $maindir/LOGS/inlist_${ivals[$inlst]}
    # $maindir/rn &> $outfl
    cp $maindir/inlist_test_tmplt $maindir/inlist_test
    sed -i 's/emom_norm_/'${ivals[$inlst]}'/g' $maindir/inlist_test
    $maindir/re $xphoto &>> LOGS/STD.out

    # # CHANGE WIMP_MODULE.f90
    # cp $maindir/src/wimp/wimp_module_$inlst.f90 $maindir/src/wimp/wimp_module.f90
    # $maindir/clean
    # $maindir/mk
    # cp $maindir/$RUNS/$xphoto $maindir/photos/.
    # $maindir/re $xphoto &>> LOGS/STD.out

    # $maindir/bash_scripts/del_dup_mods.sh $maindir &>> $outfl
    # $maindir/bash_scripts/data_reduc.sh $maindir &>> $outfl

    # pgstar movies
    # $maindir/pgstar_movie grid1
    # mv $maindir/movie.mp4 $maindir/LOGS/grid1.mp4
    # $maindir/pgstar_movie grid2
    # mv $maindir/movie.mp4 $maindir/LOGS/grid2.mp4
    # rm -r $maindir/png

    # newlogs=$maindir/$RUNS/LOGSc5_${ivals[$inlst]}
    newlogs=$maindir/$RUNS/LOGSc5_$inlst
    rm -r $newlogs
    mv $maindir/LOGS $newlogs
    mv $maindir/photos $newlogs/.
    mv $maindir/png $newlogs/.

done
