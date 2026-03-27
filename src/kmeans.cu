#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <omp.h>

namespace {

constexpr int kIntensityLevels = 256;
constexpr int kMaxK = 128;
constexpr int kMaxN = 100000;
constexpr int kMaxT = 50;
constexpr int kThreadsPerBlock = 256;
constexpr int kHistogramHashSize = 1024;
using IntensityType = std::uint8_t;
using HistogramCountType = int;


struct Point {
    int x;
    int y;
    int z;
    int intensity;
    std::string x_text;
    std::string y_text;
    std::string z_text;
};

struct InputData {
    int n = 0;
    int k = 0;
    int t = 0;
    std::vector<Point> points;
};

struct HostArrays {
    std::vector<int> xs;
    std::vector<int> ys;
    std::vector<int> zs;
    std::vector<IntensityType> intensities;
};

void cuda_check(cudaError_t status, const char* expr, const char* file, int line) {
    if (status == cudaSuccess) {
        return;
    }

    std::ostringstream oss;
    oss << "CUDA error at " << file << ':' << line << " for " << expr << ": "
        << cudaGetErrorString(status);
    throw std::runtime_error(oss.str());
}

#define CUDA_CHECK(expr) cuda_check((expr), #expr, __FILE__, __LINE__)

int parse_integer_coordinate(const std::string& text, const char axis_name, const int index) {
    std::size_t parsed = 0;
    const double value = std::stod(text, &parsed);
    if (parsed != text.size() || !std::isfinite(value)) {
        std::ostringstream oss;
        oss << "Point " << index << " has an invalid " << axis_name << " coordinate.";
        throw std::runtime_error(oss.str());
    }

    const long long rounded = std::llround(value);
    if (std::fabs(value - static_cast<double>(rounded)) > 1e-9) {
        std::ostringstream oss;
        oss << "Point " << index << " has a non-integer " << axis_name
            << " coordinate, but K-means expects integer coordinates.";
        throw std::runtime_error(oss.str());
    }
    if (rounded < std::numeric_limits<int>::min() || rounded > std::numeric_limits<int>::max()) {
        std::ostringstream oss;
        oss << "Point " << index << " has an out-of-range " << axis_name << " coordinate.";
        throw std::runtime_error(oss.str());
    }

    return static_cast<int>(rounded);
}

__host__ __device__ inline int compare_coordinates(
    const int lhs_x,
    const int lhs_y,
    const int lhs_z,
    const int rhs_x,
    const int rhs_y,
    const int rhs_z) {
    if (lhs_x < rhs_x) {
        return -1;
    }
    if (lhs_x > rhs_x) {
        return 1;
    }
    if (lhs_y < rhs_y) {
        return -1;
    }
    if (lhs_y > rhs_y) {
        return 1;
    }
    if (lhs_z < rhs_z) {
        return -1;
    }
    if (lhs_z > rhs_z) {
        return 1;
    }
    return 0;
}

__host__ __device__ inline bool better_cluster(
    const long long candidate_distance,
    const int candidate_x,
    const int candidate_y,
    const int candidate_z,
    const int candidate_cluster,
    const long long current_distance,
    const int current_x,
    const int current_y,
    const int current_z,
    const int current_cluster) {
    if (current_cluster < 0) {
        return true;
    }
    if (candidate_distance < current_distance) {
        return true;
    }
    if (candidate_distance > current_distance) {
        return false;
    }

    const int coordinate_cmp = compare_coordinates(
        candidate_x,
        candidate_y,
        candidate_z,
        current_x,
        current_y,
        current_z);
    if (coordinate_cmp != 0) {
        return coordinate_cmp < 0;
    }
    return candidate_cluster < current_cluster;
}

__host__ __device__ inline long long squared_distance(
    const int x,
    const int y,
    const int z,
    const int centroid_x,
    const int centroid_y,
    const int centroid_z) {
    const long long delta_x = static_cast<long long>(x) - centroid_x;
    const long long delta_y = static_cast<long long>(y) - centroid_y;
    const long long delta_z = static_cast<long long>(z) - centroid_z;
    return delta_x * delta_x + delta_y * delta_y + delta_z * delta_z;
}

template <typename HistogramType>
__host__ __device__ inline int remap_intensity(
    const HistogramType* histogram,
    const int original_intensity,
    const int neighborhood_size) {
    int cumulative = 0;
    int cumulative_min = 0;
    bool found_non_zero_bin = false;
    int cumulative_at_original = 0;

    for (int value = 0; value < kIntensityLevels; ++value) {
        cumulative += static_cast<int>(histogram[value]);
        if (!found_non_zero_bin && cumulative > 0) {
            cumulative_min = cumulative;
            found_non_zero_bin = true;
        }
        if (value == original_intensity) {
            cumulative_at_original = cumulative;
        }
    }

    if (!found_non_zero_bin || neighborhood_size == cumulative_min) {
        return original_intensity;
    }

    const int numerator = cumulative_at_original - cumulative_min;
    if (numerator <= 0) {
        return 0;
    }

    const int denominator = neighborhood_size - cumulative_min;
    const std::int64_t scaled = static_cast<std::int64_t>(numerator) * 255;
    const int remapped = static_cast<int>(scaled / denominator);
    if (remapped < 0) {
        return 0;
    }
    if (remapped > 255) {
        return 255;
    }
    return remapped;
}

template <int BlockSize>
__global__ void assign_clusters_kernel(
    const int* __restrict__ xs,
    const int* __restrict__ ys,
    const int* __restrict__ zs,
    const int* __restrict__ centroid_xs,
    const int* __restrict__ centroid_ys,
    const int* __restrict__ centroid_zs,
    const int n,
    const int k,
    const int* __restrict__ previous_assignments,
    int* __restrict__ next_assignments,
    int* __restrict__ changed_count) {
    __shared__ int block_changed;
    __shared__ int shared_centroid_xs[kMaxK];
    __shared__ int shared_centroid_ys[kMaxK];
    __shared__ int shared_centroid_zs[kMaxK];
    if (threadIdx.x == 0) {
        block_changed = 0;
    }
    for (int cluster = threadIdx.x; cluster < k; cluster += BlockSize) {
        shared_centroid_xs[cluster] = centroid_xs[cluster];
        shared_centroid_ys[cluster] = centroid_ys[cluster];
        shared_centroid_zs[cluster] = centroid_zs[cluster];
    }
    __syncthreads();

    const int point_index = blockIdx.x * BlockSize + threadIdx.x;
    if (point_index < n) {
        const int x = xs[point_index];
        const int y = ys[point_index];
        const int z = zs[point_index];

        long long best_distance = 0;
        int best_cluster = -1;
        int best_x = 0;
        int best_y = 0;
        int best_z = 0;

        for (int cluster = 0; cluster < k; ++cluster) {
            const int centroid_x = shared_centroid_xs[cluster];
            const int centroid_y = shared_centroid_ys[cluster];
            const int centroid_z = shared_centroid_zs[cluster];
            const long long distance = squared_distance(x, y, z, centroid_x, centroid_y, centroid_z);
            if (better_cluster(
                    distance,
                    centroid_x,
                    centroid_y,
                    centroid_z,
                    cluster,
                    best_distance,
                    best_x,
                    best_y,
                    best_z,
                    best_cluster)) {
                best_distance = distance;
                best_cluster = cluster;
                best_x = centroid_x;
                best_y = centroid_y;
                best_z = centroid_z;
            }
        }

        next_assignments[point_index] = best_cluster;
        if (previous_assignments[point_index] != best_cluster) {
            atomicAdd(&block_changed, 1);
        }
    }
    __syncthreads();
    if (threadIdx.x == 0 && block_changed > 0) {
        atomicAdd(changed_count, block_changed);
    }
}

template <int BlockSize>
__global__ void accumulate_cluster_sums_kernel(
    const int* __restrict__ xs,
    const int* __restrict__ ys,
    const int* __restrict__ zs,
    const int* __restrict__ assignments,
    const int n,
    const int k,
    int* __restrict__ sum_xs,
    int* __restrict__ sum_ys,
    int* __restrict__ sum_zs,
    int* __restrict__ counts) {
    __shared__ int block_sum_xs[kMaxK];
    __shared__ int block_sum_ys[kMaxK];
    __shared__ int block_sum_zs[kMaxK];
    __shared__ int block_counts[kMaxK];

    for (int cluster = threadIdx.x; cluster < k; cluster += BlockSize) {
        block_sum_xs[cluster] = 0;
        block_sum_ys[cluster] = 0;
        block_sum_zs[cluster] = 0;
        block_counts[cluster] = 0;
    }
    __syncthreads();

    const int point_index = blockIdx.x * BlockSize + threadIdx.x;
    if (point_index < n) {
        const int cluster = assignments[point_index];
        atomicAdd(&block_sum_xs[cluster], xs[point_index]);
        atomicAdd(&block_sum_ys[cluster], ys[point_index]);
        atomicAdd(&block_sum_zs[cluster], zs[point_index]);
        atomicAdd(&block_counts[cluster], 1);
    }
    __syncthreads();

    for (int cluster = threadIdx.x; cluster < k; cluster += BlockSize) {
        if (block_counts[cluster] != 0) {
            atomicAdd(&sum_xs[cluster], block_sum_xs[cluster]);
            atomicAdd(&sum_ys[cluster], block_sum_ys[cluster]);
            atomicAdd(&sum_zs[cluster], block_sum_zs[cluster]);
            atomicAdd(&counts[cluster], block_counts[cluster]);
        }
    }
}

template <int BlockSize>
__global__ void assign_and_accumulate_kernel(
    const int* __restrict__ xs,
    const int* __restrict__ ys,
    const int* __restrict__ zs,
    const int* __restrict__ centroid_xs,
    const int* __restrict__ centroid_ys,
    const int* __restrict__ centroid_zs,
    const int n,
    const int k,
    const int* __restrict__ previous_assignments,
    int* __restrict__ next_assignments,
    int* __restrict__ changed_count,
    int* __restrict__ sum_xs,
    int* __restrict__ sum_ys,
    int* __restrict__ sum_zs,
    int* __restrict__ counts) {
    __shared__ int block_changed;
    __shared__ int shared_centroid_xs[kMaxK];
    __shared__ int shared_centroid_ys[kMaxK];
    __shared__ int shared_centroid_zs[kMaxK];
    __shared__ int block_sum_xs[kMaxK];
    __shared__ int block_sum_ys[kMaxK];
    __shared__ int block_sum_zs[kMaxK];
    __shared__ int block_counts[kMaxK];

    if (threadIdx.x == 0) {
        block_changed = 0;
    }
    for (int cluster = threadIdx.x; cluster < k; cluster += BlockSize) {
        shared_centroid_xs[cluster] = centroid_xs[cluster];
        shared_centroid_ys[cluster] = centroid_ys[cluster];
        shared_centroid_zs[cluster] = centroid_zs[cluster];
        block_sum_xs[cluster] = 0;
        block_sum_ys[cluster] = 0;
        block_sum_zs[cluster] = 0;
        block_counts[cluster] = 0;
    }
    __syncthreads();

    const int point_index = blockIdx.x * BlockSize + threadIdx.x;
    int best_cluster = -1;
    if (point_index < n) {
        const int x = xs[point_index];
        const int y = ys[point_index];
        const int z = zs[point_index];
        long long best_distance = 0;
        int best_x = 0;
        int best_y = 0;
        int best_z = 0;

        for (int cluster = 0; cluster < k; ++cluster) {
            const int cx = shared_centroid_xs[cluster];
            const int cy = shared_centroid_ys[cluster];
            const int cz = shared_centroid_zs[cluster];
            const long long distance = squared_distance(x, y, z, cx, cy, cz);
            if (better_cluster(distance, cx, cy, cz, cluster,
                               best_distance, best_x, best_y, best_z, best_cluster)) {
                best_distance = distance;
                best_cluster = cluster;
                best_x = cx;
                best_y = cy;
                best_z = cz;
            }
        }

        next_assignments[point_index] = best_cluster;
        if (previous_assignments[point_index] != best_cluster) {
            atomicAdd(&block_changed, 1);
        }
        atomicAdd(&block_sum_xs[best_cluster], x);
        atomicAdd(&block_sum_ys[best_cluster], y);
        atomicAdd(&block_sum_zs[best_cluster], z);
        atomicAdd(&block_counts[best_cluster], 1);
    }
    __syncthreads();

    if (threadIdx.x == 0 && block_changed > 0) {
        atomicAdd(changed_count, block_changed);
    }
    for (int cluster = threadIdx.x; cluster < k; cluster += BlockSize) {
        if (block_counts[cluster] != 0) {
            atomicAdd(&sum_xs[cluster], block_sum_xs[cluster]);
            atomicAdd(&sum_ys[cluster], block_sum_ys[cluster]);
            atomicAdd(&sum_zs[cluster], block_sum_zs[cluster]);
            atomicAdd(&counts[cluster], block_counts[cluster]);
        }
    }
}

__global__ void update_centroids_kernel(
    const int k,
    const int* __restrict__ sum_xs,
    const int* __restrict__ sum_ys,
    const int* __restrict__ sum_zs,
    const int* __restrict__ counts,
    int* __restrict__ centroid_xs,
    int* __restrict__ centroid_ys,
    int* __restrict__ centroid_zs) {
    const int cluster = blockIdx.x * blockDim.x + threadIdx.x;
    if (cluster >= k) {
        return;
    }

    const int count = counts[cluster];
    if (count > 0) {
        centroid_xs[cluster] = sum_xs[cluster] / count;
        centroid_ys[cluster] = sum_ys[cluster] / count;
        centroid_zs[cluster] = sum_zs[cluster] / count;
    }
}

template <int BlockSize>
__global__ void build_cluster_histograms_kernel(
    const IntensityType* __restrict__ intensities,
    const int* __restrict__ assignments,
    const int n,
    int* __restrict__ cluster_sizes,
    HistogramCountType* __restrict__ cluster_histograms) {
    const int point_index = blockIdx.x * BlockSize + threadIdx.x;
    if (point_index >= n) {
        return;
    }
    const int cluster = assignments[point_index];
    const int intensity = static_cast<int>(intensities[point_index]);
    atomicAdd(&cluster_sizes[cluster], 1);
    atomicAdd(&cluster_histograms[cluster * kIntensityLevels + intensity], 1);
}

template <int BlockSize>
__global__ void remap_clusters_kernel(
    const IntensityType* __restrict__ intensities,
    const int* __restrict__ assignments,
    const int* __restrict__ cluster_sizes,
    const HistogramCountType* __restrict__ cluster_histograms,
    const int n,
    int* __restrict__ output_intensities) {
    const int point_index = blockIdx.x * BlockSize + threadIdx.x;
    if (point_index >= n) {
        return;
    }

    const int cluster = assignments[point_index];
    const HistogramCountType* histogram =
        cluster_histograms + cluster * kIntensityLevels;
    output_intensities[point_index] = remap_intensity(
        histogram,
        static_cast<int>(intensities[point_index]),
        cluster_sizes[cluster]);
}

InputData read_input(const std::string& path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("Failed to open input file: " + path);
    }

    InputData data;
    if (!(input >> data.n >> data.k >> data.t)) {
        throw std::runtime_error("Failed to read n, k, and T from input file.");
    }

    if (data.n <= 0) {
        throw std::runtime_error("n must be positive.");
    }
    if (data.n > kMaxN) {
        std::ostringstream oss;
        oss << "n must be at most " << kMaxN << '.';
        throw std::runtime_error(oss.str());
    }
    if (data.k <= 0) {
        throw std::runtime_error("k must be positive.");
    }
    if (data.k > kMaxK) {
        std::ostringstream oss;
        oss << "k must be at most " << kMaxK << " for this implementation.";
        throw std::runtime_error(oss.str());
    }
    if (data.k > data.n) {
        throw std::runtime_error(
            "k must be at most n because K-means uses k clusters over n points.");
    }
    if (data.t <= 0) {
        throw std::runtime_error("T must be positive.");
    }
    if (data.t > kMaxT) {
        std::ostringstream oss;
        oss << "T must be at most " << kMaxT << '.';
        throw std::runtime_error(oss.str());
    }

    data.points.reserve(data.n);
    for (int index = 0; index < data.n; ++index) {
        Point point;
        if (!(input >> point.x_text >> point.y_text >> point.z_text >> point.intensity)) {
            std::ostringstream oss;
            oss << "Failed to read point " << index << '.';
            throw std::runtime_error(oss.str());
        }

        point.x = parse_integer_coordinate(point.x_text, 'x', index);
        point.y = parse_integer_coordinate(point.y_text, 'y', index);
        point.z = parse_integer_coordinate(point.z_text, 'z', index);
        if (point.intensity < 0 || point.intensity >= kIntensityLevels) {
            std::ostringstream oss;
            oss << "Point " << index << " has intensity outside [0, 255].";
            throw std::runtime_error(oss.str());
        }

        data.points.push_back(std::move(point));
    }

    return data;
}

HostArrays build_host_arrays(const InputData& data) {
    HostArrays arrays;
    arrays.xs.resize(data.n);
    arrays.ys.resize(data.n);
    arrays.zs.resize(data.n);
    arrays.intensities.resize(data.n);

    #pragma omp parallel for schedule(static)
    for (int index = 0; index < data.n; ++index) {
        arrays.xs[index] = data.points[index].x;
        arrays.ys[index] = data.points[index].y;
        arrays.zs[index] = data.points[index].z;
        arrays.intensities[index] = static_cast<IntensityType>(data.points[index].intensity);
    }

    return arrays;
}

int assign_cluster_cpu(
    const int x,
    const int y,
    const int z,
    const std::vector<int>& centroid_xs,
    const std::vector<int>& centroid_ys,
    const std::vector<int>& centroid_zs,
    const int k) {
    long long best_distance = 0;
    int best_cluster = -1;
    int best_x = 0;
    int best_y = 0;
    int best_z = 0;

    for (int cluster = 0; cluster < k; ++cluster) {
        const int centroid_x = centroid_xs[cluster];
        const int centroid_y = centroid_ys[cluster];
        const int centroid_z = centroid_zs[cluster];
        const long long distance = squared_distance(x, y, z, centroid_x, centroid_y, centroid_z);
        if (better_cluster(
                distance,
                centroid_x,
                centroid_y,
                centroid_z,
                cluster,
                best_distance,
                best_x,
                best_y,
                best_z,
                best_cluster)) {
            best_distance = distance;
            best_cluster = cluster;
            best_x = centroid_x;
            best_y = centroid_y;
            best_z = centroid_z;
        }
    }

    return best_cluster;
}

std::vector<int> compute_kmeans_cpu(const InputData& data, const HostArrays& arrays) {
    std::vector<int> centroid_xs(data.k);
    std::vector<int> centroid_ys(data.k);
    std::vector<int> centroid_zs(data.k);
    for (int cluster = 0; cluster < data.k; ++cluster) {
        centroid_xs[cluster] = arrays.xs[cluster];
        centroid_ys[cluster] = arrays.ys[cluster];
        centroid_zs[cluster] = arrays.zs[cluster];
    }

    std::vector<int> assignments(data.n, -1);
    std::vector<int> next_assignments(data.n, -1);

    for (int iteration = 0; iteration < data.t; ++iteration) {
        int changed_count = 0;
        #pragma omp parallel for reduction(+:changed_count) schedule(static)
        for (int point_index = 0; point_index < data.n; ++point_index) {
            const int cluster = assign_cluster_cpu(
                arrays.xs[point_index],
                arrays.ys[point_index],
                arrays.zs[point_index],
                centroid_xs,
                centroid_ys,
                centroid_zs,
                data.k);
            next_assignments[point_index] = cluster;
            changed_count += (cluster != assignments[point_index]);
        }

        assignments.swap(next_assignments);
        if (changed_count == 0) {
            break;
        }

        std::vector<int> sum_xs(data.k, 0);
        std::vector<int> sum_ys(data.k, 0);
        std::vector<int> sum_zs(data.k, 0);
        std::vector<int> counts(data.k, 0);

        #pragma omp parallel
        {
            std::vector<int> local_sum_xs(data.k, 0);
            std::vector<int> local_sum_ys(data.k, 0);
            std::vector<int> local_sum_zs(data.k, 0);
            std::vector<int> local_counts(data.k, 0);

            #pragma omp for schedule(static)
            for (int point_index = 0; point_index < data.n; ++point_index) {
                const int cluster = assignments[point_index];
                local_sum_xs[cluster] += arrays.xs[point_index];
                local_sum_ys[cluster] += arrays.ys[point_index];
                local_sum_zs[cluster] += arrays.zs[point_index];
                local_counts[cluster] += 1;
            }

            #pragma omp critical
            {
                for (int cluster = 0; cluster < data.k; ++cluster) {
                    sum_xs[cluster] += local_sum_xs[cluster];
                    sum_ys[cluster] += local_sum_ys[cluster];
                    sum_zs[cluster] += local_sum_zs[cluster];
                    counts[cluster] += local_counts[cluster];
                }
            }
        }

        for (int cluster = 0; cluster < data.k; ++cluster) {
            if (counts[cluster] > 0) {
                centroid_xs[cluster] = sum_xs[cluster] / counts[cluster];
                centroid_ys[cluster] = sum_ys[cluster] / counts[cluster];
                centroid_zs[cluster] = sum_zs[cluster] / counts[cluster];
            }
        }
    }

    std::vector<int> cluster_sizes(data.k, 0);
    std::vector<HistogramCountType> cluster_histograms(
        static_cast<std::size_t>(data.k) * kIntensityLevels,
        0);
    for (int point_index = 0; point_index < data.n; ++point_index) {
        const int cluster = assignments[point_index];
        ++cluster_sizes[cluster];
        ++cluster_histograms[cluster * kIntensityLevels + arrays.intensities[point_index]];
    }

    std::vector<int> output(data.n, 0);
    #pragma omp parallel for schedule(static)
    for (int point_index = 0; point_index < data.n; ++point_index) {
        const int cluster = assignments[point_index];
        output[point_index] = remap_intensity(
            cluster_histograms.data() + cluster * kIntensityLevels,
            static_cast<int>(arrays.intensities[point_index]),
            cluster_sizes[cluster]);
    }

    return output;
}

std::vector<int> compute_kmeans_gpu(const InputData& data, const HostArrays& arrays) {
    int* d_xs = nullptr;
    int* d_ys = nullptr;
    int* d_zs = nullptr;
    IntensityType* d_intensities = nullptr;
    int* d_centroid_xs = nullptr;
    int* d_centroid_ys = nullptr;
    int* d_centroid_zs = nullptr;
    int* d_assignments_prev = nullptr;
    int* d_assignments_next = nullptr;
    int* d_changed_count = nullptr;
    int* d_sum_xs = nullptr;
    int* d_sum_ys = nullptr;
    int* d_sum_zs = nullptr;
    int* d_counts = nullptr;
    int* d_cluster_sizes = nullptr;
    HistogramCountType* d_cluster_histograms = nullptr;
    int* d_output_intensities = nullptr;

    const std::size_t point_bytes = static_cast<std::size_t>(data.n) * sizeof(int);
    const std::size_t intensity_bytes = static_cast<std::size_t>(data.n) * sizeof(IntensityType);
    const std::size_t centroid_bytes = static_cast<std::size_t>(data.k) * sizeof(int);
    const std::size_t assignment_bytes = static_cast<std::size_t>(data.n) * sizeof(int);
    const std::size_t cluster_histogram_bytes =
        static_cast<std::size_t>(data.k) * kIntensityLevels * sizeof(HistogramCountType);
    const std::size_t cluster_size_bytes = static_cast<std::size_t>(data.k) * sizeof(int);
    const std::size_t output_bytes = static_cast<std::size_t>(data.n) * sizeof(int);
    const int blocks = (data.n + kThreadsPerBlock - 1) / kThreadsPerBlock;

    std::vector<int> centroid_xs(data.k);
    std::vector<int> centroid_ys(data.k);
    std::vector<int> centroid_zs(data.k);
    for (int cluster = 0; cluster < data.k; ++cluster) {
        centroid_xs[cluster] = arrays.xs[cluster];
        centroid_ys[cluster] = arrays.ys[cluster];
        centroid_zs[cluster] = arrays.zs[cluster];
    }

    try {
        CUDA_CHECK(cudaMalloc(&d_xs, point_bytes));
        CUDA_CHECK(cudaMalloc(&d_ys, point_bytes));
        CUDA_CHECK(cudaMalloc(&d_zs, point_bytes));
        CUDA_CHECK(cudaMalloc(&d_intensities, intensity_bytes));
        CUDA_CHECK(cudaMalloc(&d_centroid_xs, centroid_bytes));
        CUDA_CHECK(cudaMalloc(&d_centroid_ys, centroid_bytes));
        CUDA_CHECK(cudaMalloc(&d_centroid_zs, centroid_bytes));
        CUDA_CHECK(cudaMalloc(&d_assignments_prev, assignment_bytes));
        CUDA_CHECK(cudaMalloc(&d_assignments_next, assignment_bytes));
        CUDA_CHECK(cudaMalloc(&d_changed_count, sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_sum_xs, centroid_bytes));
        CUDA_CHECK(cudaMalloc(&d_sum_ys, centroid_bytes));
        CUDA_CHECK(cudaMalloc(&d_sum_zs, centroid_bytes));
        CUDA_CHECK(cudaMalloc(&d_counts, centroid_bytes));
        CUDA_CHECK(cudaMalloc(&d_cluster_sizes, cluster_size_bytes));
        CUDA_CHECK(cudaMalloc(&d_cluster_histograms, cluster_histogram_bytes));
        CUDA_CHECK(cudaMalloc(&d_output_intensities, output_bytes));

        CUDA_CHECK(cudaMemcpy(d_xs, arrays.xs.data(), point_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_ys, arrays.ys.data(), point_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_zs, arrays.zs.data(), point_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(
            d_intensities,
            arrays.intensities.data(),
            intensity_bytes,
            cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(
            d_centroid_xs,
            centroid_xs.data(),
            centroid_bytes,
            cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(
            d_centroid_ys,
            centroid_ys.data(),
            centroid_bytes,
            cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(
            d_centroid_zs,
            centroid_zs.data(),
            centroid_bytes,
            cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_assignments_prev, 0xFF, assignment_bytes));

        int* final_assignments = d_assignments_prev;
        bool converged = false;

        const int centroid_blocks = (data.k + kThreadsPerBlock - 1) / kThreadsPerBlock;
        for (int iteration = 0; iteration < data.t; ++iteration) {
            CUDA_CHECK(cudaMemset(d_changed_count, 0, sizeof(int)));
            CUDA_CHECK(cudaMemset(d_sum_xs, 0, centroid_bytes));
            CUDA_CHECK(cudaMemset(d_sum_ys, 0, centroid_bytes));
            CUDA_CHECK(cudaMemset(d_sum_zs, 0, centroid_bytes));
            CUDA_CHECK(cudaMemset(d_counts, 0, centroid_bytes));
            assign_and_accumulate_kernel<kThreadsPerBlock><<<blocks, kThreadsPerBlock>>>(
                d_xs,
                d_ys,
                d_zs,
                d_centroid_xs,
                d_centroid_ys,
                d_centroid_zs,
                data.n,
                data.k,
                d_assignments_prev,
                d_assignments_next,
                d_changed_count,
                d_sum_xs,
                d_sum_ys,
                d_sum_zs,
                d_counts);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());

            int changed_count = 0;
            CUDA_CHECK(cudaMemcpy(
                &changed_count,
                d_changed_count,
                sizeof(int),
                cudaMemcpyDeviceToHost));

            final_assignments = d_assignments_next;
            if (changed_count == 0) {
                converged = true;
                break;
            }

            update_centroids_kernel<<<centroid_blocks, kThreadsPerBlock>>>(
                data.k,
                d_sum_xs,
                d_sum_ys,
                d_sum_zs,
                d_counts,
                d_centroid_xs,
                d_centroid_ys,
                d_centroid_zs);
            CUDA_CHECK(cudaGetLastError());

            std::swap(d_assignments_prev, d_assignments_next);
            final_assignments = d_assignments_prev;
        }

        if (!converged) {
            final_assignments = d_assignments_prev;
        }

        CUDA_CHECK(cudaMemset(d_cluster_sizes, 0, cluster_size_bytes));
        CUDA_CHECK(cudaMemset(d_cluster_histograms, 0, cluster_histogram_bytes));
        build_cluster_histograms_kernel<kThreadsPerBlock><<<blocks, kThreadsPerBlock>>>(
            d_intensities,
            final_assignments,
            data.n,
            d_cluster_sizes,
            d_cluster_histograms);
        CUDA_CHECK(cudaGetLastError());

        remap_clusters_kernel<kThreadsPerBlock><<<blocks, kThreadsPerBlock>>>(
            d_intensities,
            final_assignments,
            d_cluster_sizes,
            d_cluster_histograms,
            data.n,
            d_output_intensities);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<int> output(data.n, 0);
        CUDA_CHECK(cudaMemcpy(
            output.data(),
            d_output_intensities,
            output_bytes,
            cudaMemcpyDeviceToHost));

        CUDA_CHECK(cudaFree(d_xs));
        CUDA_CHECK(cudaFree(d_ys));
        CUDA_CHECK(cudaFree(d_zs));
        CUDA_CHECK(cudaFree(d_intensities));
        CUDA_CHECK(cudaFree(d_centroid_xs));
        CUDA_CHECK(cudaFree(d_centroid_ys));
        CUDA_CHECK(cudaFree(d_centroid_zs));
        CUDA_CHECK(cudaFree(d_assignments_prev));
        CUDA_CHECK(cudaFree(d_assignments_next));
        CUDA_CHECK(cudaFree(d_changed_count));
        CUDA_CHECK(cudaFree(d_sum_xs));
        CUDA_CHECK(cudaFree(d_sum_ys));
        CUDA_CHECK(cudaFree(d_sum_zs));
        CUDA_CHECK(cudaFree(d_counts));
        CUDA_CHECK(cudaFree(d_cluster_sizes));
        CUDA_CHECK(cudaFree(d_cluster_histograms));
        CUDA_CHECK(cudaFree(d_output_intensities));

        return output;
    } catch (...) {
        cudaFree(d_xs);
        cudaFree(d_ys);
        cudaFree(d_zs);
        cudaFree(d_intensities);
        cudaFree(d_centroid_xs);
        cudaFree(d_centroid_ys);
        cudaFree(d_centroid_zs);
        cudaFree(d_assignments_prev);
        cudaFree(d_assignments_next);
        cudaFree(d_changed_count);
        cudaFree(d_sum_xs);
        cudaFree(d_sum_ys);
        cudaFree(d_sum_zs);
        cudaFree(d_counts);
        cudaFree(d_cluster_sizes);
        cudaFree(d_cluster_histograms);
        cudaFree(d_output_intensities);
        throw;
    }
}

void write_output(
    const std::string& path,
    const InputData& data,
    const std::vector<int>& output_intensities) {
    std::ofstream output(path);
    if (!output) {
        throw std::runtime_error("Failed to open output file: " + path);
    }

    for (int index = 0; index < data.n; ++index) {
        output << data.points[index].x_text << ' '
               << data.points[index].y_text << ' '
               << data.points[index].z_text << ' '
               << output_intensities[index] << '\n';
    }
}

int validate_against_cpu(
    const InputData& data,
    const HostArrays& arrays,
    const std::vector<int>& gpu_output) {
    const std::vector<int> cpu_output = compute_kmeans_cpu(data, arrays);

    std::size_t mismatch_count = 0;
    for (int index = 0; index < data.n; ++index) {
        if (gpu_output[index] != cpu_output[index]) {
            if (mismatch_count < 10) {
                std::cerr << "Mismatch at point " << index
                          << ": GPU=" << gpu_output[index]
                          << ", CPU=" << cpu_output[index] << '\n';
            }
            ++mismatch_count;
        }
    }

    if (mismatch_count == 0) {
        std::cout << "Validation passed: GPU output matches CPU reference for all "
                  << data.n << " points.\n";
        return 0;
    }

    std::cerr << "Validation failed: " << mismatch_count
              << " point(s) differ from the CPU reference.\n";
    return 1;
}

void print_usage(const char* program_name) {
    std::cerr << "Usage: " << program_name
              << " [--validate] [--cpu-reference] <input_file> [output_file]\n";
}

}  // namespace

int main(int argc, char** argv) {
    try {
        bool validate = false;
        bool cpu_reference = false;
        std::vector<std::string> positional;

        for (int arg_index = 1; arg_index < argc; ++arg_index) {
            const std::string argument = argv[arg_index];
            if (argument == "--validate") {
                validate = true;
            } else if (argument == "--cpu-reference") {
                cpu_reference = true;
            } else if (argument == "--help" || argument == "-h") {
                print_usage(argv[0]);
                return 0;
            } else {
                positional.push_back(argument);
            }
        }

        if (positional.empty() || positional.size() > 2) {
            print_usage(argv[0]);
            return 1;
        }

        const std::string input_path = positional[0];
        const std::string output_path =
            positional.size() == 2 ? positional[1] : "kmeans.txt";

        const InputData data = read_input(input_path);
        const HostArrays arrays = build_host_arrays(data);
        const std::vector<int> output_intensities = cpu_reference
            ? compute_kmeans_cpu(data, arrays)
            : compute_kmeans_gpu(data, arrays);

        write_output(output_path, data, output_intensities);

        if (validate && !cpu_reference) {
            return validate_against_cpu(data, arrays, output_intensities);
        }

        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
