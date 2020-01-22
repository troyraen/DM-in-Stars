#!/usr/local/bin/bash

function dwnld () {
	dirO=$1 # LOGS directory on Osiris
	dirR=$2 # local dir
	dwnld_profs=${3:-0} # = 1 downloads profiles, default 0
	dwnld_movies=${4:-0} # = 1 downloads movies grid1.mp4 and grid2.mp4

	server="tjr63@osiris-inode01.phyast.pitt.edu"
	hdat="history.data"
	histO="${dirO}${hdat}"
	histR="${dirR}${hdat}"

    if ssh $server test -e $histO ; then # if histO exists on Osiris
	    mkdir -p "${dirR}"
        rsync -vhe ssh ${server}:${dirO}\{${hdat},controls1.data,profiles.index\} ${dirR}
		# mv ${dirR}history_reduc.data ${dirR}history.data
        if [ "${dwnld_profs}" = 1 ]; then # download profile data
            rsync -vhe ssh ${server}:${dirO}\{profile0.data,profile1.data,profile2.data,profile3.data,profile4.data,profile5.data,profiles.index\} ${dirR}
        fi
		if [ "${dwnld_movies}" = 1 ]; then # download grid1.mp4 and grid2.mp4
            rsync -vhe ssh ${server}:${dirO}\{grid1.mp4,grid2.mp4\} ${dirR}
        fi
    fi


}



maindirR="/Users/troyraen/Google_Drive/MESA/code/DATA/mesaruns/pgstarGrid2"
maindirO="/home/tjr63/mesaruns/RUNS_pgstarGrid2"
dwnld_profs=1
dwnld_movies=0


declare -A mvals=( [m0p8]=0.8 [m1p0]=1.0 [m1p1]=1.1 [m1p2]=1.2 [m1p3]=1.3 [m1p4]=1.4 [m1p6]=1.6 [m2p0]=2.0 [m3p0]=3.0 [m4p0]=4.0 )
declare -a mord=( m0p8 m1p0 m1p1 m1p2 m1p3 m1p4 m1p6 m2p0 m3p0 m4p0 )

for cb in 0 3 6; do
    for mass in "${mord[@]}"; do
		dirO="${maindirO}/c${cb}/${mass}/LOGS/"
		dirR="${maindirR}/mass${mass:1:3}/LOGSc${cb}/"
		dwnld ${dirO} ${dirR} ${dwnld_profs} ${dwnld_movies}
    done
done

#
#
# spin=SD
# # for i in {0..9}; do
# # for cb in {0..6}; do
# cb=0
# 	# for mr in {0..1}; do
# 	mr=1
# 		for mp in {2..9}; do
# 		# for mp in 83 87 92 94 96 98; do
# 	# mp=5
# 			dirO="${maindirO}/${spin}/c${cb}/m${mr}p${mp}/LOGS/"
# 			# dirO="${maindirO}/m${mr}p${mp}/LOGS/"
# 			# dirR="${maindirR}/mass${mr}p${mp}/LOGSc${cb}/"
# 			dirR="${maindirR}/mass${mr}p${mp}/LOGSc${cb}_isochrones_Aug/"
# 			# dwnld_profs=$(( $mr < 2 ? 1 : 0))
# 			dwnld_profs=0
# 			dwnld_movies=1
#
# 			dwnld ${dirO} ${dirR} ${dwnld_profs} ${dwnld_movies}
#
# 		done
# 	# done
# # done
# # done
