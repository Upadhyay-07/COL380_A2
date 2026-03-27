#!/bin/bash

echo "Running knn..."
time ./a2 input.txt knn

echo -e "\nRunning approx_knn..."
time ./a2 input.txt approx_knn

echo -e "\nRunning kmeans..."
time ./a2 input.txt kmeans

python3 mae_loss.py knn.txt approx_knn.txt