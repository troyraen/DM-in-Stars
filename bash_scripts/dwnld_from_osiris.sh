#!/usr/local/bin/bash

server="tjr63@osiris-inode01.phyast.pitt.edu"

maindirR="mesaruns"
maindir="histdat"
spin=SD

# for i in {0..6}; do
for cb in {1..6}; do
	for mr in {0..1}; do
		for mp in {8..9}; do

	dirO="/home/tjr63/${maindir}/RUNS_xL_lt_0p2/${spin}/c${cb}/m${mr}p${mp}/LOGS/"
	# histO="${dirO}history.data"
	hO="history.data"
	histO="${dirO}${hO}"
	dirR="/Users/troyraen/Google_Drive/MESA/code/DATA/${maindirR}/mass${mr}p${mp}/LOGSc${cb}/"
	histR="${dirR}history.data"

    if ssh $server test -e $histO ; then # if histO exists on Osiris
    mkdir -p "$dirR"
        rsync -vhe ssh ${server}:${dirO}\{${hO},controls1.data,profiles.index\} ${dirR}
		# mv ${dirR}history_reduc.data ${dirR}history.data
        if [ $mr -lt 2 ]; then # download profile data
            rsync -vhe ssh ${server}:${dirO}\{profile0.data,profile1.data,profile2.data,profile3.data,profile4.data,profile5.data,profile1001.data,profiles.index\} ${dirR}
        fi
    fi

		done
	done
done
# done
