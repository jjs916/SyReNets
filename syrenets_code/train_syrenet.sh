#!/bin/sh
python train.py math2 -sl 2 -d 2 -niter 1 -lambda_2 0.001
python train.py math2 -sl 2 -d 2 -niter 1 -lambda_2 0.01
python train.py math2 -sl 2 -d 2 -niter 1 -lambda_2 0.1
python train.py math2 -sl 2 -d 2 -niter 1 -lambda_2 1
