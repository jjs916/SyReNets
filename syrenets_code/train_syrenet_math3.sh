#!/bin/sh
python train.py math3 -sl 2 -d 2 -niter 20000 -lambda_2 0.001
python train.py math3 -sl 2 -d 2 -niter 20000 -lambda_2 0.01
python train.py math3 -sl 2 -d 2 -niter 20000 -lambda_2 0.1
python train.py math3 -sl 2 -d 2 -niter 20000 -lambda_2 1
python train.py math3 -sl 2 -d 4 -niter 50000 -lambda_2 0.001
python train.py math3 -sl 2 -d 4 -niter 50000 -lambda_2 0.01
python train.py math3 -sl 2 -d 4 -niter 50000 -lambda_2 0.1
python train.py math3 -sl 2 -d 4 -niter 50000 -lambda_2 1
