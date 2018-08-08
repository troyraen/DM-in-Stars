##!/usr/local/bin/bash
#!/bin/bash



######
#   This function takes a directory as input
#   generates a new file dir/history_reduc.data from dir/history.data
#   keeping only:
#       model_number == 1
#       model has profile saved
#       model_number % 5 == 0 from MS
#       num_keep total models from post-MS
#
#   ASSUMES center_h1 IS COLUMN 30 AND model_number IS COLUMN 1 OF HISTORY FILE
######
function data_reduc {
    if [ $# -eq 0 ]
      then
        echo "*********** This function requires a directory as input ***********"
        exit 1
    fi

    num_keep=150 # number of models to keep from postMS (not exact since rounding happens later)
    dir=$1
    pdat=$dir/profiles.index
    hdat=$dir/history.data
    hreduc=$dir/history_reduc.data
    hdata=$dir/hdata.txt

    lnct=$(( $(sed -n '$=' $hdat) -6 )) # get hdat line count -6 header rows
    (head -6 > $hreduc; tail -$lnct > $hdata) < $hdat # split history.data -> hheader.txt, hdata.txt
    lnct=$(sed -n '$=' $hdata) # get hdata line count for awk
    pmodarr=($(awk '{ if (NR > 1) { print $1 } }' $pdat)) # get model numbers from profiles.index
    # keep all models in profiles.index,
    # model_number%5==0 for those in MS,
    # 150 models from post MS
    awk -v nk=$num_keep -v lnct=$lnct -v pmod="${pmodarr[*]}" '
    BEGIN { nth=0; split(pmod,pm," "); for (i in pm) mod[pm[i]]=pm[i] }
    { if ($1 == 1) { print $0 }
    else if ($1 in mod) { print $0 }
    else if ($30 > 0.01) { if ($1 % 5 == 0) { print $0; } }
    else {
        if (nth == 0) {
            lnpostMS = lnct - NR
            nth = int(lnpostMS / nk)
        }
        if (nth < 3) { if ($1 % 5 == 0) { print $0 } }
        else if (NR % nth == 0) { print $0 }
    }
    }' < $hdata >> $hreduc

    rm $hdata
    echo '*** Data reduction complete ***'
    wc -l $hdat
    wc -l $hreduc
    echo
}
############


drs=($(find -name 'LOGS' -type d))
for dir in "${drs[@]}"; do
#       echo $dir
        data_reduc $dir
done
