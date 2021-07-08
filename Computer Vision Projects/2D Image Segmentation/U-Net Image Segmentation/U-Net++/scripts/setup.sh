#!/bin/bash

gdown --id 19M9Vi9ZZ7EKXHITFii9QkFDI1rB4UGOP

mkdir inputs
cd ./inputs/ && tar -xvzf ./U-Net_DataSet.tar.gz && cd ..

mkdir inputs/U-Net_Train_Deformed
