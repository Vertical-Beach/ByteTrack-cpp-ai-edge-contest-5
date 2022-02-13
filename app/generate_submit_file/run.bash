#!/bin/bash

function exit_with_invalid_args () {
    echo "Usage 1: $0 <videos directory path> <video name prefix> json <detection results directory path>"
    echo "Usage 2: $0 <videos directory path> <video name prefix> dpu <modelconfig .prototxt> <modelfile .xmodel>"
    exit 1
}

function build () {
    mkdir -p $1/build && cd $1/build
    cmake .. $2
    make -j
    cd $1
}

if [[ $# < 3 ]]; then
    exit_with_invalid_args
fi

video_dir=$1
prefix=$2
mode=$3

script_dir=`cd $(dirname $0); pwd`
video_list=`find $video_dir -type f | grep "\.avi"`

if [ $mode == "json" ]; then
    if [ $# -ne 4 ]; then
        exit_with_invalid_args
    fi
    build $script_dir ""
    for video in $video_list; do
        time $script_dir/build/generate_submit_file $video $mode $4
    done
elif [ $mode == "dpu" ]; then
    if [ $# -ne 5 ]; then
        exit_with_invalid_args
    fi
    build $script_dir "-DRISCV=ON -DDPU=ON"
    for video in $video_list; do
        time $script_dir/build/generate_submit_file $video $mode $4 $5
    done
else
    echo "Invalid mode is specified: $mode"
    exit_with_invalid_args
fi

python3 $script_dir/combine_submit.py $prefix
ls -1 | grep -E "^prediction\_$prefix\_[0-9]{2}\.json$" | xargs rm
