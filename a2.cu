#include <cuda_runtime.h>
#include <omp.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <algorithm>
#include <climits>

// ============================================================
// Config
// ============================================================
#define BLOCK_SIZE    256
#define TILE_SIZE     256   // must equal BLOCK_SIZE for tiled KNN
#define MAX_K         1024  // maximum k supported for heap in local memory
#define APPROX_MAX_R  3     // maximum grid search radius for approx KNN

#define CUDA_CHECK(e) do {                                              \
    cudaError_t _ce = (e);                                              \
    if (_ce != cudaSuccess) {                                           \
        fprintf(stderr, "CUDA error at %s:%d: %s\n",                   \
                __FILE__, __LINE__, cudaGetErrorString(_ce));           \
        exit(1);                                                        \
    }                                                                   \
} while(0)

// ============================================================
// Device helpers
// ============================================================

__device__ __forceinline__
long long sq_dist(int ax, int ay, int az, int bx, int by, int bz) {
    long long dx = ax - bx, dy = ay - by, dz = az - bz;
    return dx*dx + dy*dy + dz*dz;
}

// Returns true if point a is a "worse" neighbor than b
// (farther away; or equal distance but lex-larger coordinate)
__device__ __forceinline__
bool is_worse(long long da, int ax, int ay, int az,
              long long db, int bx, int by, int bz) {
    if (da != db) return da > db;
    if (ax != bx) return ax > bx;
    if (ay != by) return ay > by;
    return az > bz;
}

// -------------------------------------------------------
// Max-heap for k-nearest (root = worst current neighbor)
// Arrays: hd (dist), hx,hy,hz (coords), hI (intensity)
// -------------------------------------------------------
__device__ void h_swap(long long* hd, int* hx, int* hy, int* hz, int* hI,
                        int a, int b) {
    long long td = hd[a]; hd[a] = hd[b]; hd[b] = td;
    int tx = hx[a]; hx[a] = hx[b]; hx[b] = tx;
    int ty = hy[a]; hy[a] = hy[b]; hy[b] = ty;
    int tz = hz[a]; hz[a] = hz[b]; hz[b] = tz;
    int ti = hI[a]; hI[a] = hI[b]; hI[b] = ti;
}

__device__ void h_sift_down(long long* hd, int* hx, int* hy, int* hz, int* hI,
                              int sz, int i) {
    while (true) {
        int best = i, l = 2*i+1, r = 2*i+2;
        if (l < sz && is_worse(hd[l],hx[l],hy[l],hz[l],
                                hd[best],hx[best],hy[best],hz[best])) best = l;
        if (r < sz && is_worse(hd[r],hx[r],hy[r],hz[r],
                                hd[best],hx[best],hy[best],hz[best])) best = r;
        if (best == i) break;
        h_swap(hd, hx, hy, hz, hI, i, best);
        i = best;
    }
}

__device__ void h_sift_up(long long* hd, int* hx, int* hy, int* hz, int* hI,
                            int i) {
    while (i > 0) {
        int p = (i-1)/2;
        if (is_worse(hd[i],hx[i],hy[i],hz[i], hd[p],hx[p],hy[p],hz[p])) {
            h_swap(hd, hx, hy, hz, hI, i, p);
            i = p;
        } else break;
    }
}

// Insert candidate (d, px, py, pz, pI) into max-heap of capacity k
__device__ void h_insert(long long* hd, int* hx, int* hy, int* hz, int* hI,
                          int& sz, int k,
                          long long d, int px, int py, int pz, int pI) {
    if (sz < k) {
        hd[sz]=d; hx[sz]=px; hy[sz]=py; hz[sz]=pz; hI[sz]=pI;
        h_sift_up(hd, hx, hy, hz, hI, sz);
        sz++;
    } else if (is_worse(hd[0],hx[0],hy[0],hz[0], d,px,py,pz)) {
        hd[0]=d; hx[0]=px; hy[0]=py; hz[0]=pz; hI[0]=pI;
        h_sift_down(hd, hx, hy, hz, hI, k, 0);
    }
}

// -------------------------------------------------------
// Histogram equalization (per-point, device)
// hist[0..255]: counts over (self + neighbors)
// m: denominator per spec (k for KNN, cluster_size for kmeans)
// -------------------------------------------------------
__device__ int equalize(int orig_I, const int* hist, int m) {
    int cdf = 0, cdf_at_I = 0, ci_min = -1;
    for (int v = 0; v < 256; v++) {
        cdf += hist[v];
        if (ci_min < 0 && cdf > 0) ci_min = cdf;
        if (v == orig_I) cdf_at_I = cdf;
    }
    if (ci_min < 0 || m <= ci_min) return orig_I;
    double ratio = (double)(cdf_at_I - ci_min) / (double)(m - ci_min);
    int v2 = (int)floor(ratio * 255.0);
    return v2 < 0 ? 0 : (v2 > 255 ? 255 : v2);
}

// ============================================================
// Exact KNN Kernel
// One thread per query; threads in a block cooperate to load
// tiles of database points into shared memory.
// Clarification 4: histogram over self + k neighbors; m = k.
// ============================================================
__global__ void knn_kernel(
    const int* __restrict__ d_x,
    const int* __restrict__ d_y,
    const int* __restrict__ d_z,
    const int* __restrict__ d_I,
    int n, int k, int* d_new_I)
{
    __shared__ int sh_x[TILE_SIZE], sh_y[TILE_SIZE],
                   sh_z[TILE_SIZE], sh_I[TILE_SIZE];

    int qi = blockIdx.x * blockDim.x + threadIdx.x;
    bool active = (qi < n);

    // Per-thread heap in local (CUDA thread-private) memory
    long long hd[MAX_K];
    int hx[MAX_K], hy[MAX_K], hz[MAX_K], hI[MAX_K];
    int hsz = 0;

    int qx=0, qy=0, qz=0, qI=0;
    if (active) { qx=d_x[qi]; qy=d_y[qi]; qz=d_z[qi]; qI=d_I[qi]; }

    for (int ts = 0; ts < n; ts += TILE_SIZE) {
        // Cooperatively load tile into shared memory
        int t_idx = ts + (int)threadIdx.x;
        if (t_idx < n) {
            sh_x[threadIdx.x] = d_x[t_idx];
            sh_y[threadIdx.x] = d_y[t_idx];
            sh_z[threadIdx.x] = d_z[t_idx];
            sh_I[threadIdx.x] = d_I[t_idx];
        }
        __syncthreads();

        if (active) {
            int tile_end = min(TILE_SIZE, n - ts);
            for (int t = 0; t < tile_end; t++) {
                int j = ts + t;
                if (j == qi) continue;   // skip self
                long long d = sq_dist(qx, qy, qz, sh_x[t], sh_y[t], sh_z[t]);
                h_insert(hd, hx, hy, hz, hI, hsz, k,
                         d, sh_x[t], sh_y[t], sh_z[t], sh_I[t]);
            }
        }
        __syncthreads();
    }

    if (!active) return;

    // Build histogram over self (1) + k nearest neighbors = k+1 total
    int hist[256];
    for (int v = 0; v < 256; v++) hist[v] = 0;
    hist[qI]++;
    for (int j = 0; j < hsz; j++) hist[hI[j]]++;

    d_new_I[qi] = equalize(qI, hist, k);   // m = k per spec
}

void run_knn(const int* d_x, const int* d_y, const int* d_z, const int* d_I,
             int n, int k, int* d_new_I)
{
    if (k > MAX_K) {
        fprintf(stderr, "Error: k=%d exceeds MAX_K=%d. Recompile with larger MAX_K.\n",
                k, MAX_K);
        exit(1);
    }
    dim3 block(BLOCK_SIZE);
    dim3 grid((n + BLOCK_SIZE - 1) / BLOCK_SIZE);
    knn_kernel<<<grid, block>>>(d_x, d_y, d_z, d_I, n, k, d_new_I);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

// ============================================================
// Approximate KNN — 3D Grid Hashing
// Grid is built on CPU (OpenMP); queries run on GPU.
// ============================================================

__global__ void approx_knn_query_kernel(
    const int* __restrict__ sorted_x,
    const int* __restrict__ sorted_y,
    const int* __restrict__ sorted_z,
    const int* __restrict__ sorted_I,
    const int* __restrict__ sorted_orig_idx,   // original point index in sorted order
    const int* __restrict__ cell_start,
    const int* __restrict__ cell_end,
    const int* __restrict__ d_x,               // original arrays for query coords
    const int* __restrict__ d_y,
    const int* __restrict__ d_z,
    const int* __restrict__ d_I,
    int n, int k,
    int min_x, int min_y, int min_z,
    int delta, int Gx, int Gy, int Gz,
    int* d_new_I)
{
    int qi = blockIdx.x * blockDim.x + threadIdx.x;
    if (qi >= n) return;

    int qx = d_x[qi], qy = d_y[qi], qz = d_z[qi], qI = d_I[qi];
    int cx = (qx - min_x) / delta;
    int cy = (qy - min_y) / delta;
    int cz = (qz - min_z) / delta;

    long long hd[MAX_K];
    int hx[MAX_K], hy[MAX_K], hz[MAX_K], hI_a[MAX_K];
    int hsz = 0;

    for (int R = 1; R <= APPROX_MAX_R; R++) {
        for (int dcx = -R; dcx <= R; dcx++) {
            for (int dcy = -R; dcy <= R; dcy++) {
                for (int dcz = -R; dcz <= R; dcz++) {
                    // Process only new shell (|max(|dcx|,|dcy|,|dcz|)| == R)
                    if (abs(dcx) < R && abs(dcy) < R && abs(dcz) < R) continue;
                    int ncx = cx+dcx, ncy = cy+dcy, ncz = cz+dcz;
                    if (ncx<0||ncx>=Gx||ncy<0||ncy>=Gy||ncz<0||ncz>=Gz) continue;
                    int cid = ncx*Gy*Gz + ncy*Gz + ncz;
                    int st = cell_start[cid], en = cell_end[cid];
                    for (int idx = st; idx < en; idx++) {
                        if (sorted_orig_idx[idx] == qi) continue; // skip self
                        long long d = sq_dist(qx, qy, qz,
                                             sorted_x[idx], sorted_y[idx], sorted_z[idx]);
                        h_insert(hd, hx, hy, hz, hI_a, hsz, k,
                                 d, sorted_x[idx], sorted_y[idx], sorted_z[idx],
                                 sorted_I[idx]);
                    }
                }
            }
        }
        if (hsz >= k) break;
    }

    int hist[256];
    for (int v = 0; v < 256; v++) hist[v] = 0;
    hist[qI]++;
    for (int j = 0; j < hsz; j++) hist[hI_a[j]]++;

    d_new_I[qi] = equalize(qI, hist, k);
}

void run_approx_knn(const int* d_x, const int* d_y, const int* d_z, const int* d_I,
                    const int* h_x, const int* h_y, const int* h_z, const int* h_I,
                    int n, int k, int* d_new_I)
{
    if (k > MAX_K) {
        fprintf(stderr, "Error: k=%d exceeds MAX_K=%d.\n", k, MAX_K);
        exit(1);
    }

    // --- CPU: compute bounding box with OpenMP ---
    int min_x = INT_MAX, max_x = INT_MIN;
    int min_y = INT_MAX, max_y = INT_MIN;
    int min_z = INT_MAX, max_z = INT_MIN;
    #pragma omp parallel for reduction(min:min_x,min_y,min_z) reduction(max:max_x,max_y,max_z)
    for (int i = 0; i < n; i++) {
        if (h_x[i] < min_x) min_x = h_x[i];
        if (h_x[i] > max_x) max_x = h_x[i];
        if (h_y[i] < min_y) min_y = h_y[i];
        if (h_y[i] > max_y) max_y = h_y[i];
        if (h_z[i] < min_z) min_z = h_z[i];
        if (h_z[i] > max_z) max_z = h_z[i];
    }

    // Choose delta so 3x3x3 neighborhood has ~k candidates on average
    double vol = (double)(max_x - min_x + 1) *
                 (double)(max_y - min_y + 1) *
                 (double)(max_z - min_z + 1);
    int delta = (int)ceil(cbrt((double)k * vol / (27.0 * n)));
    if (delta < 1) delta = 1;

    int Gx = (max_x - min_x) / delta + 1;
    int Gy = (max_y - min_y) / delta + 1;
    int Gz = (max_z - min_z) / delta + 1;

    // Clamp total cells to avoid OOM
    long long num_cells = (long long)Gx * Gy * Gz;
    const long long MAX_CELLS = 20000000LL;
    if (num_cells > MAX_CELLS) {
        delta = (int)ceil(cbrt(vol / (double)MAX_CELLS));
        if (delta < 1) delta = 1;
        Gx = (max_x - min_x) / delta + 1;
        Gy = (max_y - min_y) / delta + 1;
        Gz = (max_z - min_z) / delta + 1;
        num_cells = (long long)Gx * Gy * Gz;
    }

    // Compute cell IDs per point
    std::vector<int> cell_ids(n);
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        int cx = (h_x[i] - min_x) / delta;
        int cy = (h_y[i] - min_y) / delta;
        int cz = (h_z[i] - min_z) / delta;
        cell_ids[i] = cx * Gy * Gz + cy * Gz + cz;
    }

    // Sort original indices by cell ID
    std::vector<int> order(n);
    for (int i = 0; i < n; i++) order[i] = i;
    std::sort(order.begin(), order.end(),
              [&](int a, int b){ return cell_ids[a] < cell_ids[b]; });

    // Build sorted coordinate/intensity/orig_idx arrays
    std::vector<int> sx(n), sy(n), sz_v(n), sI(n), sorig(n);
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        int oi = order[i];
        sx[i]    = h_x[oi];
        sy[i]    = h_y[oi];
        sz_v[i]  = h_z[oi];
        sI[i]    = h_I[oi];
        sorig[i] = oi;
    }

    // Build cell_start / cell_end  (empty cell: start=n, end=0 → loop won't fire)
    std::vector<int> cs(num_cells, n), ce(num_cells, 0);
    for (int i = 0; i < n; i++) {
        int cid = cell_ids[order[i]];
        if (i == 0 || cell_ids[order[i-1]] != cid) cs[cid] = i;
        ce[cid] = i + 1;
    }

    // Copy to GPU
    int *d_sx, *d_sy, *d_sz, *d_sI, *d_sorig, *d_cs, *d_ce;
    CUDA_CHECK(cudaMalloc(&d_sx,    (size_t)n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_sy,    (size_t)n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_sz,    (size_t)n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_sI,    (size_t)n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_sorig, (size_t)n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_cs,    (size_t)num_cells * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_ce,    (size_t)num_cells * sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_sx,    sx.data(),    n*sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_sy,    sy.data(),    n*sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_sz,    sz_v.data(),  n*sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_sI,    sI.data(),    n*sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_sorig, sorig.data(), n*sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cs, cs.data(), num_cells*sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ce, ce.data(), num_cells*sizeof(int), cudaMemcpyHostToDevice));

    dim3 block(BLOCK_SIZE);
    dim3 grid((n + BLOCK_SIZE - 1) / BLOCK_SIZE);
    approx_knn_query_kernel<<<grid, block>>>(
        d_sx, d_sy, d_sz, d_sI, d_sorig,
        d_cs, d_ce,
        d_x, d_y, d_z, d_I,
        n, k,
        min_x, min_y, min_z,
        delta, Gx, Gy, Gz,
        d_new_I);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaFree(d_sx); cudaFree(d_sy); cudaFree(d_sz); cudaFree(d_sI); cudaFree(d_sorig);
    cudaFree(d_cs); cudaFree(d_ce);
}

// ============================================================
// K-Means
// ============================================================

// Assign each point to its nearest centroid
__global__ void kmeans_assign(
    const int* __restrict__ d_x,
    const int* __restrict__ d_y,
    const int* __restrict__ d_z,
    const int* __restrict__ d_cx,
    const int* __restrict__ d_cy,
    const int* __restrict__ d_cz,
    int* d_assign, int n, int k, int* d_changed)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    int px = d_x[i], py = d_y[i], pz = d_z[i];
    long long best_dist = LLONG_MAX;
    int best_c = 0;

    for (int c = 0; c < k; c++) {
        long long d = sq_dist(px, py, pz, d_cx[c], d_cy[c], d_cz[c]);
        bool better;
        if (d != best_dist) {
            better = (d < best_dist);
        } else {
            // Tie-break: lex-smaller centroid wins
            if      (d_cx[c] != d_cx[best_c]) better = (d_cx[c] < d_cx[best_c]);
            else if (d_cy[c] != d_cy[best_c]) better = (d_cy[c] < d_cy[best_c]);
            else                               better = (d_cz[c] < d_cz[best_c]);
        }
        if (better) { best_dist = d; best_c = c; }
    }
    if (d_assign[i] != best_c) {
        d_assign[i] = best_c;
        atomicOr(d_changed, 1);
    }
}

// Accumulate coordinate sums and counts per cluster
__global__ void kmeans_accum(
    const int* __restrict__ d_x,
    const int* __restrict__ d_y,
    const int* __restrict__ d_z,
    const int* __restrict__ d_assign,
    long long* d_sum_x, long long* d_sum_y, long long* d_sum_z,
    int* d_count, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int c = d_assign[i];
    atomicAdd((unsigned long long*)&d_sum_x[c], (unsigned long long)(long long)d_x[i]);
    atomicAdd((unsigned long long*)&d_sum_y[c], (unsigned long long)(long long)d_y[i]);
    atomicAdd((unsigned long long*)&d_sum_z[c], (unsigned long long)(long long)d_z[i]);
    atomicAdd(&d_count[c], 1);
}

// Update centroids: integer division (Clarification 5)
__global__ void kmeans_update(
    int* d_cx, int* d_cy, int* d_cz,
    const long long* d_sum_x, const long long* d_sum_y, const long long* d_sum_z,
    const int* d_count, int k)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= k || d_count[c] == 0) return;
    d_cx[c] = (int)(d_sum_x[c] / (long long)d_count[c]);
    d_cy[c] = (int)(d_sum_y[c] / (long long)d_count[c]);
    d_cz[c] = (int)(d_sum_z[c] / (long long)d_count[c]);
}

// Compute per-cluster histograms and cluster sizes
__global__ void kmeans_hist(
    const int* __restrict__ d_I,
    const int* __restrict__ d_assign,
    int* d_cluster_hist,    // k * 256
    int* d_cluster_size,
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int c = d_assign[i];
    atomicAdd(&d_cluster_hist[c * 256 + d_I[i]], 1);
    atomicAdd(&d_cluster_size[c], 1);
}

// Remap intensities using cluster histogram; m = cluster_size
__global__ void kmeans_remap(
    const int* __restrict__ d_I,
    const int* __restrict__ d_assign,
    const int* __restrict__ d_cluster_hist,
    const int* __restrict__ d_cluster_size,
    int* d_new_I, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int c    = d_assign[i];
    int orig = d_I[i];
    int m    = d_cluster_size[c];
    const int* hist = d_cluster_hist + c * 256;
    d_new_I[i] = equalize(orig, hist, m);
}

void run_kmeans(const int* d_x, const int* d_y, const int* d_z, const int* d_I,
                const int* h_x, const int* h_y, const int* h_z,
                int n, int k, int T, int* d_new_I)
{
    // Initialize centroids = first k points
    int *d_cx, *d_cy, *d_cz;
    CUDA_CHECK(cudaMalloc(&d_cx, k * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_cy, k * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_cz, k * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_cx, h_x, k*sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cy, h_y, k*sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cz, h_z, k*sizeof(int), cudaMemcpyHostToDevice));

    int *d_assign;
    CUDA_CHECK(cudaMalloc(&d_assign, n * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_assign, 0xFF, n * sizeof(int)));  // -1

    long long *d_sum_x, *d_sum_y, *d_sum_z;
    int *d_count, *d_changed;
    CUDA_CHECK(cudaMalloc(&d_sum_x,   k * sizeof(long long)));
    CUDA_CHECK(cudaMalloc(&d_sum_y,   k * sizeof(long long)));
    CUDA_CHECK(cudaMalloc(&d_sum_z,   k * sizeof(long long)));
    CUDA_CHECK(cudaMalloc(&d_count,   k * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_changed, sizeof(int)));

    dim3 blk(BLOCK_SIZE);
    dim3 gn((n + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 gk((k + BLOCK_SIZE - 1) / BLOCK_SIZE);

    for (int iter = 0; iter < T; iter++) {
        CUDA_CHECK(cudaMemset(d_changed, 0, sizeof(int)));
        kmeans_assign<<<gn, blk>>>(d_x, d_y, d_z, d_cx, d_cy, d_cz,
                                    d_assign, n, k, d_changed);
        CUDA_CHECK(cudaGetLastError());

        int h_changed = 0;
        CUDA_CHECK(cudaMemcpy(&h_changed, d_changed, sizeof(int), cudaMemcpyDeviceToHost));
        if (!h_changed) break;

        CUDA_CHECK(cudaMemset(d_sum_x, 0, k * sizeof(long long)));
        CUDA_CHECK(cudaMemset(d_sum_y, 0, k * sizeof(long long)));
        CUDA_CHECK(cudaMemset(d_sum_z, 0, k * sizeof(long long)));
        CUDA_CHECK(cudaMemset(d_count, 0, k * sizeof(int)));
        kmeans_accum<<<gn, blk>>>(d_x, d_y, d_z, d_assign,
                                   d_sum_x, d_sum_y, d_sum_z, d_count, n);
        CUDA_CHECK(cudaGetLastError());
        kmeans_update<<<gk, blk>>>(d_cx, d_cy, d_cz,
                                    d_sum_x, d_sum_y, d_sum_z, d_count, k);
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // Compute cluster histograms then remap intensities
    int *d_ch, *d_cs_size;
    CUDA_CHECK(cudaMalloc(&d_ch,      (size_t)k * 256 * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_cs_size, k * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_ch,      0, (size_t)k * 256 * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_cs_size, 0, k * sizeof(int)));

    kmeans_hist<<<gn, blk>>>(d_I, d_assign, d_ch, d_cs_size, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    kmeans_remap<<<gn, blk>>>(d_I, d_assign, d_ch, d_cs_size, d_new_I, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaFree(d_cx); cudaFree(d_cy); cudaFree(d_cz);
    cudaFree(d_assign);
    cudaFree(d_sum_x); cudaFree(d_sum_y); cudaFree(d_sum_z);
    cudaFree(d_count); cudaFree(d_changed);
    cudaFree(d_ch); cudaFree(d_cs_size);
}

// ============================================================
// I/O
// ============================================================
void read_input(const char* path, int& n, int& k, int& T,
                std::vector<int>& hx, std::vector<int>& hy,
                std::vector<int>& hz, std::vector<int>& hI)
{
    FILE* f = fopen(path, "r");
    if (!f) { fprintf(stderr, "Cannot open input: %s\n", path); exit(1); }
    if (fscanf(f, "%d %d %d", &n, &k, &T) != 3) {
        fprintf(stderr, "Bad header in %s\n", path); exit(1);
    }
    hx.resize(n); hy.resize(n); hz.resize(n); hI.resize(n);
    for (int i = 0; i < n; i++) {
        if (fscanf(f, "%d %d %d %d", &hx[i], &hy[i], &hz[i], &hI[i]) != 4) {
            fprintf(stderr, "Bad point at index %d\n", i); exit(1);
        }
    }
    fclose(f);
}

void write_output(const char* path,
                  const std::vector<int>& hx, const std::vector<int>& hy,
                  const std::vector<int>& hz, const std::vector<int>& new_I,
                  int n)
{
    FILE* f = fopen(path, "w");
    if (!f) { fprintf(stderr, "Cannot open output: %s\n", path); exit(1); }
    for (int i = 0; i < n; i++)
        fprintf(f, "%d %d %d %d\n", hx[i], hy[i], hz[i], new_I[i]);
    fclose(f);
}

// ============================================================
// main
// ============================================================
int main(int argc, char* argv[])
{
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <input.txt> <knn|approx_knn|kmeans>\n", argv[0]);
        return 1;
    }

    int n, k, T;
    std::vector<int> hx, hy, hz, hI;
    read_input(argv[1], n, k, T, hx, hy, hz, hI);

    // Allocate device memory
    int *d_x, *d_y, *d_z, *d_I, *d_new_I;
    CUDA_CHECK(cudaMalloc(&d_x,     (size_t)n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_y,     (size_t)n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_z,     (size_t)n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_I,     (size_t)n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_new_I, (size_t)n * sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_x, hx.data(), n*sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y, hy.data(), n*sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_z, hz.data(), n*sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_I, hI.data(), n*sizeof(int), cudaMemcpyHostToDevice));

    std::string mode(argv[2]);
    const char* out_file = nullptr;

    if (mode == "knn") {
        out_file = "knn.txt";
        run_knn(d_x, d_y, d_z, d_I, n, k, d_new_I);

    } else if (mode == "approx_knn") {
        out_file = "approx_knn.txt";
        run_approx_knn(d_x, d_y, d_z, d_I,
                       hx.data(), hy.data(), hz.data(), hI.data(),
                       n, k, d_new_I);

    } else if (mode == "kmeans") {
        out_file = "kmeans.txt";
        run_kmeans(d_x, d_y, d_z, d_I,
                   hx.data(), hy.data(), hz.data(),
                   n, k, T, d_new_I);

    } else {
        fprintf(stderr, "Unknown mode: %s (use knn, approx_knn, or kmeans)\n", argv[2]);
        return 1;
    }

    std::vector<int> h_new_I(n);
    CUDA_CHECK(cudaMemcpy(h_new_I.data(), d_new_I, n*sizeof(int),
                          cudaMemcpyDeviceToHost));
    write_output(out_file, hx, hy, hz, h_new_I, n);

    cudaFree(d_x); cudaFree(d_y); cudaFree(d_z);
    cudaFree(d_I); cudaFree(d_new_I);
    return 0;
}
