#!/usr/local/bin/bash

maindir=/Users/troyraen/Google_Drive/MESA/code/DATA/mesaruns/mass1p0
stout=($(find ${maindir} -name 'STD.out'))
for fl in "${stout[@]}"; do
    echo $fl
    tail -10 $fl
    echo
done
