##!/usr/local/bin/bash
#!/bin/bash

# Purpose of this script is to move (copy) files on Osiris
# to match a directory structure as expected by my programs
# when accessing files on Roy.
#

function move_on_O () {
	olddir=$1 # move from here
	newdir=$2 # move to here
	move_profs=${3:-0} # = 1 downloads profiles, default 0
	move_movies=${4:-0} # = 1 downloads movies grid1.mp4 and grid2.mp4

	hdat="history.data"
	histOld="${olddir}${hdat}"
	histNew="${newdir}${hdat}"
	# echo "Checking for ${histOld}"

	if [ -f "${histOld}" ]; then # if histOld exists
		echo "Moving ${histOld} to ${histNew}"
	    mkdir -p "${newdir}"
		# rsync -vh ${olddir}\{${hdat},controls1.data,profiles.index\} ${newdir}
        rsync -vh ${histOld} ${newdir}
		rsync -vh "${olddir}/controls1.data" ${newdir}
		rsync -vh "${olddir}/profiles.index" ${newdir}

        if [ "${move_profs}" = 1 ]; then # move profile data
            rsync -vh ${olddir}\{profile0.data,profile1.data,profile2.data,profile3.data,profile4.data,profile5.data,profiles.index\} ${newdir}
        fi
		if [ "${move_movies}" = 1 ]; then # move grid1.mp4 and grid2.mp4
            rsync -vh ${olddir}\{grid1.mp4,grid2.mp4\} ${newdir}
        fi
    fi


}


# basedir="/Users/troyraen/Osiris/mesaruns/RUNS_2test_final" # Run script on Roy. Assumes Osiris has been remote mounted to Roy.
basedir="/home/tjr63/mesaruns/RUNS_2test_final" # Run script on Osiris.
newbasedir="${basedir}/plotsdata"
move_profs=0
move_movies=0

for cb in $(seq 0 6); do
# cb=0
	mdrs=($(find "${basedir}/c${cb}" -name 'm*p*' -type d))
	for massdir in "${mdrs[@]}"; do
		mr="${massdir: -4:1}"
		mp="${massdir: -2:2}"
		newdir="${newbasedir}/mass${mr}p${mp}/LOGSc${cb}"
		# echo
		# echo "${massdir}"
		# echo "${newdir}"
		move_on_O "${massdir}/LOGS/" "${newdir}/" "${move_profs}" "${move_movies}"
	done
done
