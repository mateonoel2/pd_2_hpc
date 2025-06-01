#!/bin/bash
for p in 2 3 4 5 6 7 8; do
    mpirun -n $p python ejercicio_3/knn_benchmark.py --data-multiplier 2 --no-plot
done
