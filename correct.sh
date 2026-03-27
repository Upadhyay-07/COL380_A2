#!/bin/bash

# Run commands
./a2 input1.txt knn
./a2 input1.txt kmeans

# Compare outputs
echo "Diff for knn:"
diff knn.txt knn1.txt

echo "Diff for kmeans:"
diff kmeans.txt kmeans1.txt

# Cleanup
rm -f knn.txt kmeans.txt