#!/bin/bash



export MESA_DIR=/home/tjr63/mesa-r10398
export OMP_NUM_THREADS=1
export MESA_BASE=/home/tjr63/mesaruns
# !!! If you change MESA_BASE you must change the file paths in inlist and condor_wrapper !!!


export MESA_INLIST=$MESA_BASE/inlist
rnmesa=rnMESAn
export MESA_RUN=$MESA_BASE/RUNS_dmEmoment_condor/${rnmesa}
logfile=$MESA_BASE/batch_run/logs/${rnmesa}.out
cp $MESA_BASE/batch_run/${rnmesa}.sh $MESA_BASE/batch_run/logs/.
