##!/usr/local/bin/bash
#!/bin/bash

######
# This script takes a directory as input
######### CHECK WHETHER dir SHOULD INCLUDE "LOGS" ########
# Copies dir/LOGS/history.data to dir/LOGS/history_pre_del_dup_mods.data
# Generates a new history.data file stripped of duplicate models
# written because of MESA backups or restarts
# (keeping only last line for any duplicated model numbers)
######

if [ $# -eq 0 ]
  then
    echo "*********** Must supply script with dir for dir/LOGS/history.data file to be cleaned ***********"
    exit 1
fi

# dir=$1
dir=$1/LOGS
hdat=$dir/history.data
hdatpdd=$dir/history_pre_del_dup_mods.data
cp -n $hdat $hdatpdd # will not overwrite existing and so current $hdat will be lost
hhead=$dir/hheader.txt
hdata=$dir/hdata.txt
smods=$dir/single_mods.txt


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
# read -p "Press enter to continue"
sort -C -u -n -k1 $smods # sort by model number, check whether this changes the file
if [ $? -eq 0 ]
then
    sort -C -u -g -k2 $smods # sort by age, check whether this changes the file
    if [ $? -eq 0 ]
    then
        cp $hhead $hdat
        cat $smods >> $hdat # write new history.data
        rm $hhead $hdata $smods # clean up
        echo "*** $hdat cleaned of duplicate models due to backups and restarts ***"
        wc -l $hdat
        wc -l $hdatpdd
    else
      echo "*********** ERROR: $smods star_age (column 2) not strictly monotonic. History Data Not Cleaned."
      mv $hdatpdd $hdat
      rm $hhead $hdata $smods # clean up
    fi
else
  echo "*********** ERROR: $smods model_number (column 1) not strictly monotonic. History Data Not Cleaned."
  mv $hdatpdd $hdat
  rm $hhead $hdata $smods # clean up
fi
