#!/usr/local/bin/bash

server="tjr63@osiris-inode01.phyast.pitt.edu"

maindir="mesaruns"
# maindir="histdat"
spin=SD
for cb in {0..6}; do
	for mr in {0..5}; do
		for mp in {0..9}; do

	dirO="/home/tjr63/${maindir}/RUNS/${spin}/c${cb}/m${mr}p${mp}/LOGS/"
	histO="${dirO}history.data"
	# histO="${dirO}history_reduc.data"
	dirR="/Users/troyraen/Google_Drive/MESA/code/DATA/${maindir}/mass${mr}p${mp}/LOGSc${cb}/"
	histR="${dirR}history.data"

    if ssh $server test -e $histO ; then # if histO exists on Osiris
    mkdir -p "$dirR"
        rsync -vhe ssh ${server}:${dirO}\{history.data,controls1.data,profiles.index\} ${dirR}
		# rsync -vhe ssh ${server}:${dirO}\{history_reduc.data,controls1.data,profiles.index\} ${dirR}
		# mv ${dirR}history_reduc.data ${dirR}history.data
        if [ $mr -lt 2 ]; then # download profile data
            rsync -vhe ssh ${server}:${dirO}\{profile0.data,profile1.data,profile2.data,profile3.data,profile4.data,profile5.data,profile1001.data,profiles.index\} ${dirR}
        fi
    fi

		done
	done
done
