##!/usr/local/bin/bash
#!/bin/bash

######
# This script takes a directory as input
# Copies dir/history.data to dir/historyOG.data
# Generates a new history.data file stripped of duplicate models
# written because of MESA backups or restarts
# (keeping only last line for any duplicated model numbers)
######

if [ $# -eq 0 ]
  then
    echo "*********** Must supply script with dir of the history.data file to be cleaned ***********"
    exit 1
fi


dir=$1
hdat='$dir/history.data'
cp $hdat $dir/historyOG.data
hhead='$dir/hheader.txt'
hdata='$dir/hdata.txt'
smods='$dir/single_mods.txt'


lnct=$(( $(sed -n '$=' $hdat) -6 )) # get hdat line count -6 header rows
(head -6 > $hhead; tail -$lnct > $hdata) < $hdat # split history.data -> hheader.txt, hdata.txt
# hdata.txt -> single_mods.txt keeping only last of any duplicated model numbers
awk '{last_dup_mod[$1] = NR; lines[$1] = $0}
END {
  for(key in last_dup_mod) reversed_ldm[last_dup_mod[key]] = key
  for(nr=1;nr<=NR;nr++)
    if(nr in reversed_ldm) print lines[reversed_ldm[nr]]
}' < $hdata > $smods
# Check that model numbers and ages are strictly monotonic
# if they are, merge header with data
read -p "Press enter to continue"
sort -C -u -n -k1 $smods # sort by model number, check whether this changes the file
if [ $? -eq 0 ]
then
    sort -C -u -g -k2 $smods # sort by age, check whether this changes the file
    if [ $? -eq 0 ]
    then
        cp $hhead $hdat
        cat $smods >> $hdat # write new history.data
        rm $hhead $hdata $smods # clean up
    else
      echo "*********** ERROR: $dir/$smods model_number (column 1) not strictly monotonic"
      exit 1
    fi
else
  echo "*********** ERROR: $dir/$smods star_age (column 2) not strictly monotonic"
  exit 1
fi
