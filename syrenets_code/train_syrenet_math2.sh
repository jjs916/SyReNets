#!/bin/sh
python train.py math2 -sl 2 -d 4 -niter 20000 -mb 1000 -lambda_2 0.001
python train.py math2 -sl 2 -d 4 -niter 20000 -mb 1000 -lambda_2 0.01
python train.py math2 -sl 2 -d 4 -niter 20000 -mb 1000 -lambda_2 0.1
python train.py math2 -sl 2 -d 4 -niter 20000 -mb 1000 -lambda_2 1
python train.py math2 -sl 2 -d 4 -niter 20000 -mb 1000 -lambda_2 0.001
python train.py math2 -sl 2 -d 4 -niter 20000 -mb 1000 -lambda_2 0.01
python train.py math2 -sl 2 -d 4 -niter 20000 -mb 1000 -lambda_2 0.1
python train.py math2 -sl 2 -d 4 -niter 20000 -mb 1000 -lambda_2 1
