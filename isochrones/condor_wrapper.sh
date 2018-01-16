#!/bin/bash

# Use to run all 10 rnIsoMass#.sh files through Condor
#/home/tjr63/mesa_wimps_4isoc/isochrones/rnIsoMasses/rnIsoMass$1.sh

# Use to make directories:
for i in {0..9}; do
        /home/tjr63/mesa_wimps_4isoc/isochrones/rnIsoMasses/rnIsoMass$i.sh
done
