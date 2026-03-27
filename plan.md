# COL380 A2 — CUDA + OpenMP Histogram Equalization Plan

## Context
Implement local histogram equalization on 3D point clouds using CUDA, via three methods:
1. Exact KNN (brute-force GPU)
2. Approximate KNN (3D grid hashing)
3. K-Means clustering

Command interface: `./a2 input.txt [knn|approx_knn|kmeans]`
Outputs: `knn.txt`, `approx_knn.txt`, `kmeans.txt`

---

## Key Decisions (confirmed with user)

### KNN neighborhood interpretation (Clarification 4)
- Find k nearest neighbors **excluding** self
- Histogram is over **k+1 points** (self + k neighbors)
- m = k (as stated in original assignment)
- Edge case: if `m <= Ci_min`, return `Ii` unchanged (guard handles k+1 histogram where all points share same intensity)

### Tie-breaking (Clarification 3)
- Equal distance → pick point with lexicographically smaller (x, y, z)
- Coordinates are integers (Clarification 1) → exact integer comparison, no epsilon needed

### Centroid updates (Clarification 5)
- Integer division: `cx = sum_x / count` (truncation toward zero)
- Remapping: use `double`, then `floor`

---

## File Structure

```
a2/
  a2.cu       — single CUDA source: main(), I/O, all three algorithms
  Makefile
```

Single file chosen to avoid multi-file nvcc linking complexity.

---

## Data Layout

**SOA (Struct of Arrays)** on GPU for coalesced memory access:
```cpp
int *d_x, *d_y, *d_z;  // integer coordinates
int *d_I;               // intensity [0..255]
```
Host side uses `std::vector<int>` arrays (same layout).

---

## Algorithm 1: Exact KNN

### CUDA Kernel: block-per-query + shared memory tiling

- Launch `N` thread blocks, `BLOCK_SIZE=256` threads each
- Block `b` handles query point `b`
- Threads cooperate to load tiles of `BLOCK_SIZE` database points into shared memory
- Each thread maintains a local **max-heap of size k** (in local/register memory) covering its stride of database points

```
For each tile of BLOCK_SIZE database points:
  - Thread t loads shared_x[t], shared_y[t], shared_z[t], shared_I[t]
  - __syncthreads()
  - Thread t iterates over all BLOCK_SIZE tile entries, computing squared dist to query
    (each thread checks ALL tile entries → all threads benefit from shared mem)
  - Each thread updates its local heap
  - __syncthreads()
```

Wait — in block-per-query, threads check *different subsets* of the tile OR all the same tile?

**Clarification**: In block-per-query, all `BLOCK_SIZE` threads handle the **same query** but split the database. Thread `t` is responsible for database points `t, t+BLOCK_SIZE, t+2*BLOCK_SIZE, ...`. The tile loading is: thread `t` loads `shared[t] = db[tile_start + t]`. All threads in block then read from shared memory for their assigned points in the tile. Since all threads read the same tile but access different entries, this minimizes global memory traffic.

After processing all tiles, each thread has a local heap of up to k candidates. Then do **block-level merge** to find the global top-k:
- All threads write their heap to shared memory: `shared_heap[t*k_per_thread .. (t+1)*k_per_thread]`
- Thread 0 does a sequential merge of all heaps (for small BLOCK_SIZE × k)
- Or: use bitonic sort on the flattened heap array in shared memory

**Heap data per thread**: `(long long dist, int x, int y, int z, int I)` — k entries
- Squared distances use `long long` to avoid overflow
- Each thread processes N/BLOCK_SIZE database points independently

**After merge**: thread 0 has the k nearest neighbors' intensities, builds 256-bin histogram, computes CDF, applies remapping, stores result in `d_new_I[qi]`

**Tie-breaking comparator** (custom less-than for heap):
```
(d1, x1, y1, z1) < (d2, x2, y2, z2) iff d1 < d2,
  or d1 == d2 && x1 < x2,
  or d1 == d2 && x1 == x2 && y1 < y2,
  or d1 == d2 && x1 == x2 && y1 == y2 && z1 < z2
```

**Histogram equalization** (device function, called per query point):
```
histogram over self + k neighbors (k+1 total)
CDF[v] = sum hist[0..v]
Ci_min = first positive CDF value
if (m <= Ci_min) return orig_I   // m = k
new_I = floor((CDF[Ii] - Ci_min) / (double)(m - Ci_min) * 255.0)
```

---

## Algorithm 2: Approximate KNN (3D Grid Hashing)

### Steps

**1. Bounding box** (GPU reduction or CPU with OpenMP):
- Find min_x, max_x, min_y, max_y, min_z, max_z

**2. Grid cell size δ**:
- Volume V = (max_x - min_x) * (max_y - min_y) * (max_z - min_z)
- δ = max(1, (int)ceil(cbrt((double)k * V / n)))  — ensures ~k points per cell on average
- Grid dims: Gx = ceil((max_x - min_x + 1) / δ), Gy, Gz (clamped to reasonable max)

**3. Assign cell IDs** (N threads):
- `cell_id[i] = ((x[i]-min_x)/δ)*Gy*Gz + ((y[i]-min_y)/δ)*Gz + ((z[i]-min_z)/δ)`

**4. Sort points by cell ID** (GPU counting/bitonic sort):
- Counting sort: count per cell → prefix sum → scatter points
- Yields `sorted_idx[]` and `cell_start[cell_id]`, `cell_end[cell_id]`

**5. Query kernel** (N threads, one per query):
- Compute query's cell (cx, cy, cz)
- Iterate over cells in a shell of radius R (start R=1, i.e. 3×3×3=27 cells)
- For each neighboring cell, iterate over its points, compute squared distance
- Maintain max-heap of size k (same as exact KNN, excluding self)
- If fewer than k candidates found in R=1 shell, expand to R=2 (5×5×5=125 cells), etc.
- Same tie-breaking + histogram equalization as exact KNN

**Counting sort implementation** (no Thrust/CUB):
1. `count_kernel`: atomicAdd into `cell_count[cell_id]`
2. `scan_kernel`: parallel prefix sum (segmented scan in shared memory, two-pass for large arrays)
3. `scatter_kernel`: use second atomic to place each point at its sorted position

---

## Algorithm 3: K-Means

### Steps

**1. Initialize centroids**: copy first k points as initial centroids (host → device memcpy)

**2. Assignment kernel** (N threads, each handles one point):
```
- Load centroid coords into shared memory (if k * 3 * sizeof(int) fits)
- For each centroid c: compute squared dist d
- Track best_c (min dist, tie-break by lex compare of centroid coords)
- If assignments[i] != best_c: set changed_flag = true; update assignments[i]
```
`changed_flag` is a device int, reset to 0 before each iteration, set via atomic.

**3. Update kernel** (two parts):
- Part A (N threads): atomicAdd into `sum_x[c]`, `sum_y[c]`, `sum_z[c]` (long long), `count[c]` (int)
- Part B (k threads): `cx[c] = (int)(sum_x[c] / count[c])` (integer division per Clarification 5)
  - If count[c] == 0, keep centroid unchanged

**4. Convergence check**: copy `changed_flag` to host each iteration. If 0, stop. Stop also after T iterations.

**5. Histogram phase** (after convergence):
- Kernel: atomic increments into `cluster_hist[c * 256 + v]` (N threads)
- `cluster_size[c]` = count of points in cluster c
- Final kernel (N threads): each point reads its cluster histogram, computes CDF, applies remapping

**Tie-breaking for equal centroid distances**:
```cpp
// Compare centroid c vs current best_c lexicographically
if (d < best_dist ||
    (d == best_dist && (cx[c] < cx[best_c] ||
    (cx[c] == cx[best_c] && cy[c] < cy[best_c]) ||
    (cx[c] == cx[best_c] && cy[c] == cy[best_c] && cz[c] < cz[best_c])))) {
    best_c = c; best_dist = d;
}
```

---

## Host-side OpenMP Usage

- **Input parsing**: read all n lines in parallel (split into chunks, parse with `#pragma omp parallel for`)
- **Output writing**: parallel write of n output lines
- OpenMP thread count: `omp_get_max_threads()`

---

## main() Structure

```cpp
int main(int argc, char* argv[]) {
    // argv[1] = input file, argv[2] = "knn" | "approx_knn" | "kmeans"

    // Read input (OpenMP parallel)
    int n, k, T;
    std::vector<int> h_x, h_y, h_z, h_I;
    read_input(argv[1], n, k, T, h_x, h_y, h_z, h_I);

    // Allocate + copy to device
    int *d_x, *d_y, *d_z, *d_I, *d_new_I;
    // ... cudaMalloc + cudaMemcpy ...

    std::string mode = argv[2];
    if (mode == "knn")        run_knn(d_x, d_y, d_z, d_I, n, k, d_new_I);
    else if (mode == "approx_knn") run_approx_knn(d_x, d_y, d_z, d_I, n, k, d_new_I);
    else if (mode == "kmeans") run_kmeans(d_x, d_y, d_z, d_I, n, k, T, d_new_I);

    // Copy results back + write output (OpenMP parallel)
    std::vector<int> h_new_I(n);
    cudaMemcpy(h_new_I.data(), d_new_I, n*sizeof(int), cudaMemcpyDeviceToHost);
    write_output(/* filename based on mode */, h_x, h_y, h_z, h_new_I, n);
}
```

Output filename: `knn.txt`, `approx_knn.txt`, `kmeans.txt` (matching Clarification 2).

---

## Makefile

```makefile
NVCC = nvcc
NVCCFLAGS = -O3 -arch=sm_75 -Xcompiler -fopenmp

all: a2

a2: a2.cu
	$(NVCC) $(NVCCFLAGS) -o a2 a2.cu -lm

clean:
	rm -f a2 knn.txt approx_knn.txt kmeans.txt
```

`-arch=sm_75` for Turing (RTX 20xx); adjust if target GPU differs.

---

## Critical Files to Create

- `/home/upadhyay/COL380_A2/a2.cu` — full implementation
- `/home/upadhyay/COL380_A2/Makefile`

---

## Verification Plan

1. `make` — should produce `./a2` with no errors
2. Generate test input via `dataset_generator.py`
3. `./a2 input.txt knn` → compare `knn.txt` against `seq_knn.txt` from `cpu_reference.cpp`
4. `./a2 input.txt kmeans` → compare `kmeans.txt` against `seq_kmeans.txt`
5. `./a2 input.txt approx_knn` → run `mae_loss.py` against `knn.txt` to check MAE
6. Test edge cases: k=1, k=N-1, all same intensities, duplicate coordinates

---

## Open Questions / Risks

1. **Shared memory per block for k-nearest heap**: if k is large (e.g. 1024), storing k-sized heaps per thread may exceed shared memory. Plan: use local memory (slower but correct).

2. **Grid overflow in approx KNN**: if point cloud is very sparse, δ may be large and grid may have few cells. If very dense, too many cells. Need clamping of G to reasonable bounds (e.g. max 1024 per dimension).

3. **Centroid tie-breaking race condition**: if multiple assignment changes happen in parallel, the `changed_flag` atomic is fine (OR-semantics). Assignment writes are safe since each thread writes to a different index.
