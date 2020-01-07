##!/usr/local/bin/bash
#!/bin/bash



######
#   This function takes a directory as input (and optionally, keep_mod)
#   generates a new file dir/history_reduc.data from dir/history.data
#   keeping only:
#       model_number == 1
#       model has profile saved
#       model_number % $keep_mod == 0 from MS
#       num_keep total models from post-MS
#
#   for loop at bottom runs function on list of LOGS dirs
#
######
function data_reduc {
    if [ $# -eq 0 ]
      then
        echo "*********** This function (data_reduc) requires a directory as input ***********"
        exit 1
    fi

    num_keep=500 # number of models to keep from postMS (not exact since rounding happens later)
    dir=$1
    keep_mod=${2:-5} # keep MS models where model_number % $keep_mod == 0
    pdat=$dir/profiles.index
    hdat=$dir/history.data
    hreduc=$dir/history_reduc.data
    hdata=$dir/hdata.txt

    lnct=$(( $(sed -n '$=' $hdat) -6 )) # get hdat line count -6 header rows
    (head -6 > $hreduc; tail -$lnct > $hdata) < $hdat # split history.data -> hheader.txt, hdata.txt
    lnct=$(sed -n '$=' $hdata) # get hdata line count for awk
    pmodarr=($(awk '{ if (NR > 1) { print $1 } }' $pdat)) # get model numbers from profiles.index
    # find model_number and center_h1 column numbers
    read mcol h1col <<< $(awk '{ if (NR == 6) {
                for(l=1;l<=NF;l++) {
                    if($l=="model_number") { m = l }
                    else if($l=="center_h1") { h1 = l }
                }}}
                END{ print m " " h1 }' < $hdat)

    # keep all models in profiles.index,
    # model_number%5==0 for those in MS,
    # num_keep models from post MS
    awk -v nk=$num_keep -v mcol=$mcol -v h1col=$h1col -v lnct=$lnct -v pmod="${pmodarr[*]}" '
    BEGIN { nth=0; split(pmod,pm," "); for (i in pm) mod[pm[i]]=pm[i] }
    { if ($mcol == 1) { print $0 }
    else if ($mcol in mod) { print $0 }
    else if ($h1col > 0.001) { if ($mcol % $keep_mod == 0) { print $0; } }
    else {
        if (nth == 0) {
            lnpostMS = lnct - NR
            nth = int(lnpostMS / nk)
        }
        if (nth < 3) { if ($mcol % $keep_mod == 0) { print $0 } }
        else if (NR % nth == 0) { print $0 }
    }
    }' < $hdata >> $hreduc

    rm $hdata
    echo '*** Data reduction complete ***'
    wc -l $hdat
    wc -l $hreduc
    echo
}

data_reduc $1/LOGS


############

# # drs=($(find /Users/troyraen/Google_Drive/MESA/code/DATA/mesaruns/mass0p8 -name 'LOGS*' -type d))
# drs=($(find /home/tjr63/mesaruns/RUNS -name 'LOGS'))
# for dir in "${drs[@]}"; do
#     # echo $dir
#     ./del_dup_mods.sh $dir
#     data_reduc $dir
#     # mv ${dir}/history_reduc.data ${dir}/history.data
#     # rm ${dir}/history_reduc.data
# done


############
# maindir=/home/tjr63/histdat/RUNS_xL_lt_0p2/SD
# drs=($(find /home/tjr63/mesaruns/RUNS -name 'c[0-6]'))
# for dir in "${drs[@]}"; do
#     # echo $dir
#     cp -r $dir ${maindir}/.
# done
# drs=($(find ${maindir} -name 'LOGS'))
# for dir in "${drs[@]}"; do
#     ./del_dup_mods.sh ${dir}
#     rm ${dir}/history_pre_del_dup_mods.data
#     data_reduc ${dir}
#     mv ${dir}/history_reduc.data ${dir}/history.data
#     # rm ${dir}/history_reduc.data
# done
