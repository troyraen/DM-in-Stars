##!/usr/local/bin/bash
#!/bin/bash

####
#   This script generates the input file isocinput for Aaron's isochrone program
#       based on history.data files in hdatadir
#   Can run ./make_eep and ./make_iso at end
#
#   Assumes duplicate models (from backups and restarts) have
#       been removed from history files
#   Assumes working directory is iso so that
#       data is in ./data/tracks/cb/
#
#   Script takes one argument: cboost number in [0..6]
###

if [ $# -eq 0 ]
  then
    echo "*********** Must supply script with cboost argument in [0..6] ***********"
    exit 1
fi

cb=$1 # script input argument
isodir="/home/tjr63/DMS/isomy"
hdatadir="data/tracks/c${cb}" # history.data files should be here and named m#p#.data
eepinput="input.eep"
isocinput="input.example" # will create this file
isocoutput="isochrone_c$cb.dat" # make_iso writes to this file

cd ${isodir}
mkdir -p data/eeps
mkdir -p data/isochrones

declare -a hfiles # store names of history files for input.example

# get the m#p## files
# mr=0
# for mp in 80 85 90 95; do
#     hdat=${hdatadir}/m${mr}p${mp}.data
#     if [ -e $hdat ]; then
#         hfiles=("${hfiles[@]}" "m${mr}p${mp}.data")
#     fi
# done

# get the other files
for mr in {0..5}; do # for loops ensure list is ordered in increasing mass
    # for mp in {0..9}; do
    for mp in $(seq 0 99); do
        if [ ${mp} -lt 10 ]; then
            mass="m${mr}p0${mp}"
    	else
    		mass="m${mr}p${mp}"
    	fi
        hdat="${hdatadir}/${mass}.data"
        if [ -e $hdat ]; then
            hfiles=("${hfiles[@]}" "${mass}.data")
        fi
    done
done

len=$(echo ${#hfiles[@]})
# cp /home/tjr63/mesaruns/history_columns.list .

echo
echo "generating $isocinput"
echo
# create input.example (overwrites existing)
echo "#version string, max 8 characters
example
#initial Y, initial Z, [Fe/H], [alpha/Fe], v/vcrit (space separated)
   0.2703 1.42e-02   0.00        0.00     0.0
#data directories: 1) history files, 2) eeps, 3) isochrones
${hdatadir}
data/eeps
data/isochrones
#read history_columns
history_columns.list
#specify tracks
${len}" > $isocinput
printf "%s\n" "${hfiles[@]}" >> $isocinput
echo "#specify isochrones
${isocoutput}
min_max
log10
51
7.0
10.21" >> $isocinput

# keep a copy of input files
cp ${isocinput} ${eepinput} ${hdatadir}/.

# uncomment these lines to run make_eep and make_iso
# export ISO_DIR=$(pwd)
./make_both $isocinput
