# Algorithm Explanation

This document explains, in words, how the current implementation solves all three parts of the assignment:

1. exact KNN based local histogram equalization
2. approximate KNN based local histogram equalization
3. K-means based local histogram equalization

It describes the algorithmic flow and the main implementation ideas, but deliberately avoids code.

## 1. Overall Structure

The program reads a point cloud of `n` unique points. Each point contains:

- an integer `x` coordinate
- an integer `y` coordinate
- an integer `z` coordinate
- an intensity `I` in `[0, 255]`

The input order is preserved throughout the computation. Every output file writes the points in the same order as the input, with only the intensity changed.

Internally, the implementation stores the coordinates and intensities in separate arrays instead of storing them as an array of structs. This layout is better for GPU execution because threads often read the same field from many points at once.

The three outputs are:

- `knn.txt` for exact KNN
- `approx_knn.txt` for approximate KNN
- `kmeans.txt` for K-means

## 2. Common Rules Followed by All Parts

### Neighborhood semantics

For the two KNN-based methods:

- the query point itself is not part of its own nearest-neighbor search
- the local histogram is still computed over the query point plus its `k` neighbors
- so the histogram size is `k + 1`

For K-means:

- a point belongs to exactly one cluster
- the cluster already includes the point itself
- so the local histogram is the histogram of the whole cluster

### Tie-breaking

Whenever two candidate points are at the same Euclidean distance, the lexicographically smaller point is preferred. Lexicographic order means:

1. compare `x`
2. if `x` ties, compare `y`
3. if `y` also ties, compare `z`

For K-means assignment, if a point is equally distant from two centroids, the lexicographically smaller centroid is chosen.

### Intensity remapping

All three methods perform local histogram equalization after the neighborhood has been decided.

For KNN and approximate KNN, the implementation does not explicitly store a full 256-bin histogram for each query point inside the GPU kernel. Instead, it computes only the information needed for the remapping formula:

- the minimum intensity in the local neighborhood
- how many times that minimum occurs
- how many neighborhood intensities are less than or equal to the center point intensity

This is enough to evaluate the same remapping result while using much less per-thread local state.

For K-means, the implementation does build a real 256-bin histogram per cluster, because many points share the same cluster histogram and reuse makes that approach worthwhile.

All remapped intensities are clipped to `[0, 255]`.

## 3. Shared Data Preparation for the KNN-Based Parts

The unified executable uses one common spatial index for both exact KNN and approximate KNN.

### Why a shared index is used

Both exact KNN and approximate KNN search for nearby points in the same 3D space. Rebuilding a separate spatial structure for each method would duplicate:

- the same cell construction work on the CPU
- the same sorting work
- the same upload of cell metadata and reordered point arrays to the GPU

So the implementation builds the KNN spatial index once and reuses it for both methods.

### How the spatial index is built

The point cloud is partitioned into grid cells in 3D space.

The cell size is chosen automatically using a sampling heuristic:

1. sample up to 1024 points spread across the input
2. for each sampled point, compute its exact `k`-th neighbor distance on the CPU
3. collect those distances
4. choose the 60th percentile of the sampled `k`-th distances as the cell size

This cell size is only a search heuristic. It does not affect exactness in the exact KNN algorithm. It only affects how efficiently nearby points are grouped into cells.

### What is stored in the index

After the cell size is chosen:

1. each point is mapped to an integer cell coordinate `(cell_x, cell_y, cell_z)`
2. all points are sorted by cell coordinate
3. the program records, for each occupied cell:
   - the cell coordinate
   - the start position of that cell in the sorted point array
   - the end position of that cell in the sorted point array

The sorted arrays and cell metadata are then uploaded to the GPU.

This shared structure is what the benchmark labels as `shared_knn`.

## 4. Exact KNN

### Main idea

The exact KNN algorithm performs an exact nearest-neighbor search using the voxel grid as a pruning structure.

It is still exact because it never stops until it can prove that no point outside the already visited cells can beat the current `k`-th best neighbor.

### Per-query process

Each GPU thread handles one query point.

For that query point:

1. compute the grid cell containing the query
2. examine neighboring cells shell by shell around the query cell
3. for every candidate point found in those cells:
   - ignore the query point itself
   - compute squared Euclidean distance
   - update the thread-local top-`k` neighbor list

The top-`k` list is kept sorted from best to worst. Because `k <= 128`, this is stored in fixed-size thread-local arrays.

### Shell expansion

The search does not inspect all cells immediately. It expands in shells:

- first the query cell and the closest surrounding region
- then progressively larger shells if needed

After each shell, the algorithm checks whether the current `k`-th best distance is already small enough that every still-unseen cell must be farther away than that.

This check uses a geometric lower bound:

- for any unseen shell outside the current radius, the program computes the minimum possible distance from the query point to that outside region
- if the current `k`-th neighbor is already closer than this minimum possible outside distance, the search is complete

At that point the method is exact, because no unvisited point can change the top-`k` result.

### Histogram equalization stage

Once the exact neighbors are known:

1. include the center point intensity
2. include the intensities of the exact `k` neighbors
3. compute the local remapping statistics
4. produce the final equalized intensity for that point

Because each thread writes its result directly to the output location corresponding to the original point index, the final file preserves input order automatically.

## 5. Approximate KNN

### Main idea

The approximate KNN algorithm uses the same spatial index, but intentionally performs a more aggressive and cheaper neighborhood search.

The current version is designed to trade a small amount of MAE for speed.

### Query strategy

For each query point, the algorithm does the following:

1. search only the query point's own cell first
2. if that cell does not provide enough candidate points, expand to the immediately surrounding shell
3. if the combined candidates are still fewer than `k`, mark the query for fallback

Unlike exact KNN, this method does not keep expanding until it proves exactness. It stops much earlier by design.

### Why this is faster

The approximate method is faster when:

- most queries find enough useful neighbors in their own cell or the first surrounding shell
- only a small fraction of points need fallback

In that regime, many expensive outer shells are never visited.

### Exact fallback

If the approximate search cannot even gather `k` candidates, the query is considered unresolved.

Those unresolved queries are compacted into a fallback list. A second GPU kernel then computes exact neighbors for only those points by scanning the whole dataset.

This fallback preserves robustness:

- easy queries use the cheap approximate path
- difficult queries use exact search only when necessary

The benchmark output includes the number of fallback queries so that this tradeoff can be measured directly.

### Histogram equalization stage

After the approximate neighbors are selected, the same local remapping logic as exact KNN is used:

- the query point intensity is included
- the `k` selected neighbors are included
- the local equalized intensity is computed from those `k + 1` intensities

So the only approximation is in neighbor selection, not in the remapping formula.

## 6. K-Means

### Main idea

The K-means part clusters the whole point cloud into `k` clusters and then uses each point's cluster as its local neighborhood.

### Initialization

The initial centroids are the first `k` points from the input, exactly as required by the assignment clarification.

Each centroid stores three integer coordinates.

### Assignment step

Each point is assigned to the nearest centroid using squared Euclidean distance.

If two centroids are tied, the lexicographically smaller centroid is chosen.

On the GPU, one thread handles one point during assignment.

To reduce repeated global memory traffic, centroid coordinates are first copied into shared memory inside each block before points are compared against them.

### Stopping rule

After every assignment step, the implementation counts how many points changed clusters.

The algorithm stops when:

- no assignment changes in an iteration, or
- the maximum number of iterations `T` is reached

### Centroid update

Once the new assignments are available:

1. sum the `x`, `y`, and `z` coordinates of all points assigned to each cluster
2. count how many points belong to each cluster
3. update each centroid with integer division:
   - `centroid_x = sum_x / count`
   - `centroid_y = sum_y / count`
   - `centroid_z = sum_z / count`

If a cluster becomes empty, its centroid is left unchanged.

### GPU implementation details

The centroid accumulation step is done in two levels:

1. each block accumulates local cluster sums and counts in shared memory
2. those block-local partial sums are then added to the global cluster sums

This reduces pressure on global atomics compared to updating global arrays directly for every point.

### Building cluster histograms

After final assignments are known, the implementation builds one histogram per cluster.

For this stage:

- each point contributes its intensity to the histogram of its assigned cluster
- cluster sizes are counted at the same time

The GPU kernel uses a block-local shared-memory hash table to reduce contention before flushing counts into the global cluster histogram array.

This is useful because the histogram space is conceptually two-dimensional:

- cluster id
- intensity value from 0 to 255

### Remapping stage

Once cluster histograms are built:

1. each point looks up its cluster
2. it uses that cluster's 256-bin histogram
3. it computes the remapped intensity using the full cluster size as `m`

This matches the assignment rule that K-means neighborhoods are complete clusters, including the center point.

## 7. CPU Reference Paths

The codebase also contains CPU reference implementations for exact KNN and K-means.

These are used for:

- validation
- correctness checks
- benchmark comparison on smaller cases

They follow the same assignment rules as the GPU versions:

- same tie-breaking
- same neighborhood definitions
- same remapping semantics
- same output ordering

Approximate KNN is compared against the exact KNN result, since that is the intended correctness reference for the approximate method.

## 8. Benchmarking Strategy

There are two benchmark modes.

### Small and medium validated benchmark

This mode runs:

- exact KNN CPU reference
- exact KNN GPU
- approximate KNN GPU
- K-means CPU reference
- K-means GPU

It is used on moderate input sizes where the sequential baselines are still practical.

### Large GPU-only benchmark

For very large cases, sequential exact KNN is too expensive to run repeatedly. So a separate GPU-only benchmark mode measures:

- exact KNN GPU
- approximate KNN GPU
- K-means GPU

This mode is used to study scaling up to `n = 100000`.

## 9. Why the Three Parts Differ

The three methods solve the same local histogram equalization problem, but they define locality differently.

### Exact KNN

- neighborhood is the true nearest-neighbor set
- highest correctness guarantee
- most search work per query

### Approximate KNN

- neighborhood is a fast estimate based on nearby grid cells
- much cheaper when local density is favorable
- may deviate slightly from exact KNN, but fallback protects difficult queries

### K-Means

- neighborhood is the entire cluster
- no per-point neighbor search
- quality depends on how well clustering reflects local structure

Because the neighborhood definitions differ, the cost structure and optimization opportunities also differ:

- exact and approximate KNN spend most of their effort on spatial search
- K-means spends most of its effort on repeated assignment and centroid update

## 10. Summary

The implementation is organized around a shared 3D spatial index for the KNN-based parts and a separate iterative clustering pipeline for K-means.

The main ideas are:

- exact KNN uses shell expansion with a geometric stopping rule to remain exact
- approximate KNN uses a shallow shell search plus exact fallback to trade a little accuracy for speed
- K-means uses iterative assignment and centroid recomputation, then cluster histograms for remapping

All three methods ultimately produce a new intensity for every input point while preserving input order and following the clarified assignment semantics.
