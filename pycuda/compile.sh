#!/bin/sh
cython -3 --cplus kernel.pyx -o kernel.cu
nvcc -O3 -std=c++14 -o kernel.so --shared --compiler-options -fPIC kernel.cu -I/cm/shared/apps/python/3.6.5/include/python3.6m/

