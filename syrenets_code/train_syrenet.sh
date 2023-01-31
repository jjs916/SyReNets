#!/bin/sh
python train.py math3 -sl 4 -d 4 -niter 10000 -lambda_2 0.001
python train.py math3 -sl 4 -d 4 -niter 10000 -lambda_2 0.01
python train.py math3 -sl 4 -d 4 -niter 10000 -lambda_2 0.1
python train.py math3 -sl 4 -d 4 -niter 10000 -lambda_2 1
