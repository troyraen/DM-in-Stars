#!/bin/bash

declare -A ivals=( [jina]=jina [mts]=mesh_timestep [opa]=opacity [ovs]=overshoot [ow]=other_wind )
declare -a iord=( opa jina mts ovs ow )


maindir=/home/tjr63/mesaruns
outfl=$maindir/LOGS/STD.out
for inlst in "${iord[@]}"; do
    mkdir $maindir/LOGS $maindir/png $maindir/photos
    cp $maindir/inlist_${ivals[$inlst]} $maindir/inlist
    cp $maindir/inlist $maindir/LOGS/inlist_${ivals[$inlst]}
    ./rn &> $outfl
    ./bash_scripts/del_dup_mods.sh $maindir &>> $outfl
    ./bash_scripts/data_reduc.sh $maindir &>> $outfl

    ./pgstar_movie grid1
    mv $maindir/movie.mp4 $maindir/LOGS/grid1.mp4
    ./pgstar_movie grid2
    mv $maindir/movie.mp4 $maindir/LOGS/grid2.mp4

    newlogs=$maindir/RUNS_refined_mist/LOGSc5_${ivals[$inlst]}
    mv $maindir/LOGS $newlogs
    mv $maindir/photos $newlogs/.
    mv $maindir/png $newlogs/.
