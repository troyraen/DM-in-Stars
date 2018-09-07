#!/bin/bash

declare -A ivals=( [jina]=jina [mts]=mesh_timestep [opa]=opacity [ovs]=overshoot [ow]=other_wind [opos]=opacity_overshoot [oth]=other [OG]=OG)
declare -a iord=( OG opa mts ow opos oth )


maindir=/home/tjr63/mesaruns
outfl=$maindir/LOGS/STD.out
for inlst in "${iord[@]}"; do
    mkdir $maindir/LOGS $maindir/png $maindir/photos
    cp $maindir/inlist_${ivals[$inlst]} $maindir/inlist
    cp $maindir/inlist $maindir/LOGS/inlist_${ivals[$inlst]}
    $maindir/rn &> $outfl
    $maindir/bash_scripts/del_dup_mods.sh $maindir &>> $outfl
    $maindir/bash_scripts/data_reduc.sh $maindir &>> $outfl

    $maindir/pgstar_movie grid1
    mv $maindir/movie.mp4 $maindir/LOGS/grid1.mp4
    $maindir/pgstar_movie grid2
    mv $maindir/movie.mp4 $maindir/LOGS/grid2.mp4

    newlogs=$maindir/RUNS_refined_mist/LOGSc5_${ivals[$inlst]}
    mv $maindir/LOGS $newlogs
    mv $maindir/photos $newlogs/.
    mv $maindir/png $newlogs/.

done
