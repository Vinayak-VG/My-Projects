#!/bin/bash

wget --no-check-certificate https://shapenet.cs.stanford.edu/media/shapenetcore_partanno_segmentation_benchmark_v0_normal.zip

mkdir inputs
cd ./inputs/ && unzip ./shapenetcore_partanno_segmentation_benchmark_v0_normal.zip && cd ..