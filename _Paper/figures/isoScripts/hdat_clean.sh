#!/usr/local/bin/bash
##!/bin/bash

######
#   Assumes working directory is iso so that
#       python script is ./scripts/hdat_clean.py
#
#   This function takes a cboost (int) as input
#   Finds all history files in mesaruns dir
#   calls a python script to generates a new file
#       newdir/history.data from olddir/history.data
#       keeping only columns in clist
#
######

if [ $# -eq 0 ]
  then
    echo "*********** Must supply script with cboost argument in [0..6] ***********"
    exit 1
fi


cb=$1
newdir="/Users/troyraen/Osiris/isomy/data/tracks/c$cb"
mkdir -p $newdir
drs=($(find /Users/troyraen/Osiris/mesaruns/RUNS_2test_final/plotsdata -name 'mass*' -type d))
for olddir in "${drs[@]}"; do
    hdatold="$olddir/LOGSc$cb/history.data"
    mass="${olddir: -3}"
    if [[ $mass == p* ]] # check to see if mass has 2 digits after 'p' (m#p##)
    then
        mass="${olddir: -4}"
    fi
    hdatnew="$newdir/m$mass.data"
    # hdatnew="$newdir/m${olddir: -3}.data"
    if [ ! -e ${hdatnew} ]; then
        echo
        echo 'processing' $hdatold
        echo 'writing to' $hdatnew
        echo
        python ./scripts/hdat_clean.py $hdatold $hdatnew
    else
        echo ${hdatnew} 'exists. Skipping...'
    fi
done


######
#   This function takes as input:
#       . cboost (int)
#       . newdir in newdir/mass.data (history.data files for isochrones input)
#           . (e.g. "/Users/troyraen/Google_Drive/MESA/code/iso/data/tracks/c$cb")
#           . will be created if it does not exist
#       . olddrs_parent in olddrs_parent/
#           . (e.g. "/Users/troyraen/Google_Drive/MESA/code/DATA/mesaruns")
#   Finds all history files in mesaruns dir
#   calls a python script to generates a new file
#       newdir/history.data from olddir/history.data
#       keeping only columns in clist
#   Assumes working directory is iso so that
#       python script is ./scripts/hdat_clean.py
#
######



#
# function data_reduc {
#     if [ $# -eq 0 ] # NEED NOT EQUAL TO 3
#       then
#         echo "*********** This function (data_reduc) requires 2 inputs ***********"
#         echo "*********** olddir and newdir for history.data files"
#         exit 1
#     fi
#
#     # num_keep=500 # number of models to keep from postMS (not exact since rounding happens later)
#     olddir=$1
#     newdir=$2
#     # pdat=$dir/profiles.index
#     hdat=history.data
#     # hreduc=$dir/history_reduc.data
#     tmpdat=$newdir/tmp.data
#
#     lnct=$(( $(sed -n '$=' $olddir/$hdat) -4 )) # get hdat line count -4 header rows
#     (head -4 > $newdir/$hdat; tail -$lnct > $tmpdat) < $olddir/$hdat # split history.data -> hheader.txt, hdata.txt
#     lnct=$(sed -n '$=' $hdata) # get hdata line count for awk
#     pmodarr=($(awk '{ if (NR > 1) { print $1 } }' $pdat)) # get model numbers from profiles.index
#     # find model_number and center_h1 column numbers
#     read mcol h1col <<< $(awk '{ if (NR == 6) {
#                 for(l=1;l<=NF;l++) {
#                     if($l=="model_number") { m = l }
#                     else if($l=="center_h1") { h1 = l }
#                 }}}
#                 END{ print m " " h1 }' < $tmpdat)
#
#     # keep all models in profiles.index,
#     # model_number%5==0 for those in MS,
#     # num_keep models from post MS
#     awk -v nk=$num_keep -v mcol=$mcol -v h1col=$h1col -v lnct=$lnct -v pmod="${pmodarr[*]}" '
#     BEGIN { nth=0; split(pmod,pm," "); for (i in pm) mod[pm[i]]=pm[i] }
#     { if ($mcol == 1) { print $0 }
#     else if ($mcol in mod) { print $0 }
#     else if ($h1col > 0.01) { if ($mcol % 5 == 0) { print $0; } }
#     else {
#         if (nth == 0) {
#             lnpostMS = lnct - NR
#             nth = int(lnpostMS / nk)
#         }
#         if (nth < 3) { if ($mcol % 5 == 0) { print $0 } }
#         else if (NR % nth == 0) { print $0 }
#     }
#     }' < $hdata >> $hreduc
#
#     rm $hdata
#     echo '*** Data reduction complete ***'
#     wc -l $hdat
#     wc -l $hreduc
#     echo
# }
#
# data_reduc $1/LOGS
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
# def extract_data_lines(filename, start_text, end_text, include_start=False,
#                        include_end=False):
#     """
#     open `filename`, and yield the lines between
#     the line that contains `start_text` and the line that contains `end_text`.
#     """
#     # started = False
#     with open(filename) as fh:
#         for line in fh:
#             yield line
#
#             # if started:
#             #     if end_text in line:
#             #         if include_end:
#             #             yield line
#             #         break
#             #     yield line
#             # elif start_text in line:
#             #     started = True
#             #     if include_start:
#             #         yield line
#
#
# get list of directories
# for each dir:
#     get history.data
#     process each line in file
#     get first 4
#     line 6: if column in ['star_age', 'star_mass']:
#
# read in to pandas DF, remove columns, write out
