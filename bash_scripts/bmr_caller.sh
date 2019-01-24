#!/bin/bash

# This script is used to call bulk_mesa_run.sh

nscreens=5

# cboost
cstart=3
cinc=1
cstop=6

# complete mass set to run
mstart=5.0
mstop=0.8
minc=-0.05

# echo $(seq ${mstart} ${minc} ${mstop})

# separate masses to run in nscreens separate screens
mstart_list=($(seq ${mstart} ${minc} ${mstop}))
mstart_list=(${mstart_list[@]:0:${nscreens}})

new_minc=$(echo $nscreens \* $minc | bc -l)
maindir="/home/tjr63/mesaruns/bash_scripts"
RUNS="newfolder"

for ms in "${mstart_list[@]}"; do
    screen -dm ./${maindir}/bulk_mesa_run.sh ${RUNS} ${ms} ${new_minc} ${mstop} ${cstart} ${cinc} ${cstop}
	# echo "list $ms"
    # echo $(seq $ms $new_minc $mstop)
	# echo
done
