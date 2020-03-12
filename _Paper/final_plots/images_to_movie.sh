#!/bin/sh
#
# Wrapper to create movies from images

set -o noglob

in_files=$1
out_file=$2

$MESASDK_ROOT/bin/ffmpeg \
    -loglevel warning \
    -pattern_type glob -i $in_files \
    -pix_fmt yuv420p \
    -vf 'crop=trunc(iw/2)*2:trunc(ih/2)*2:0:0' \
    -y \
    $out_file
