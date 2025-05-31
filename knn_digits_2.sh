#!/bin/bash
for p in 1 2 4 8; do
    mpirun -n $p python knn_digits_2.py --data-multiplier 2 --no-plot
done

for n in 5000 10000 20000 40000; do
    mpirun -n 8 python knn_digits_2.py --synthetic --n-samples $n --no-plot
done
