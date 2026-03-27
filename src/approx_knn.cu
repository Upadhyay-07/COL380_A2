#include <cuda_runtime.h>

#include <algorithm>
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
constexpr int kThreadsPerBlock = 128;
constexpr int kSampleBudget = 1024;
constexpr int kFirstShellRadius = 1;
constexpr int kSecondShellRadius = 2;
constexpr double kVoxelDistancePercentile = 0.60;
using IntensityType = std::uint8_t;
using HistogramCountType = std::uint16_t;
using FlagType = std::uint8_t;
constexpr uint64_t kCellHashMulX = 0x9e3779b97f4a7c15ULL;
constexpr uint64_t kCellHashMulY = 0xc2b2ae3d27d4eb4fULL;
constexpr uint64_t kCellHashMulZ = 0x165667b19e3779f9ULL;

struct Point {
    double x;
    double y;
    double z;
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
    std::vector<double> xs;
    std::vector<double> ys;
    std::vector<double> zs;
    std::vector<IntensityType> intensities;
};

struct CellPoint {
    long long cell_x;
    long long cell_y;
    long long cell_z;
    int original_index;
};

struct SpatialIndex {
    double cell_size = 1.0;
    double inv_cell_size = 1.0;
    std::vector<long long> cell_xs;
    std::vector<long long> cell_ys;
    std::vector<long long> cell_zs;
    std::vector<int> cell_starts;
    std::vector<int> cell_ends;
    std::vector<double> sorted_xs;
    std::vector<double> sorted_ys;
    std::vector<double> sorted_zs;
    std::vector<IntensityType> sorted_intensities;
    std::vector<int> sorted_original_indices;
    std::vector<int> cell_hash_values;
    int cell_hash_mask = 0;
};

struct ApproxResult {
    std::vector<int> output_intensities;
    double cell_size = 1.0;
    int num_cells = 0;
    int fallback_count = 0;
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

__host__ __device__ inline int compare_coordinates(
    const double lhs_x,
    const double lhs_y,
    const double lhs_z,
    const double rhs_x,
    const double rhs_y,
    const double rhs_z) {
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

__host__ __device__ inline bool better_candidate(
    const double candidate_distance,
    const double candidate_x,
    const double candidate_y,
    const double candidate_z,
    const int candidate_index,
    const double current_distance,
    const int current_index,
    const double* current_xs,
    const double* current_ys,
    const double* current_zs) {
    if (current_index < 0) {
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
        current_xs[current_index],
        current_ys[current_index],
        current_zs[current_index]);
    if (coordinate_cmp != 0) {
        return coordinate_cmp < 0;
    }
    return candidate_index < current_index;
}

template <typename HistogramType>
__host__ __device__ inline int remap_intensity_from_neighbors(
    const IntensityType center_intensity,
    const IntensityType* neighbor_intensities,
    const int neighbor_count) {
    int min_intensity = static_cast<int>(center_intensity);
    int count_min = 1;
    int count_leq_center = 1;

    for (int neighbor = 0; neighbor < neighbor_count; ++neighbor) {
        const int intensity = static_cast<int>(neighbor_intensities[neighbor]);
        if (intensity < min_intensity) {
            min_intensity = intensity;
            count_min = 1;
        } else if (intensity == min_intensity) {
            ++count_min;
        }
        if (intensity <= static_cast<int>(center_intensity)) {
            ++count_leq_center;
        }
    }

    const int neighborhood_size = neighbor_count + 1;
    if (neighborhood_size == count_min) {
        return static_cast<int>(center_intensity);
    }

    const int numerator = count_leq_center - count_min;
    if (numerator <= 0) {
        return 0;
    }

    const int denominator = neighborhood_size - count_min;
    const std::int64_t scaled = static_cast<std::int64_t>(numerator) * 255;
    int remapped = static_cast<int>(scaled / denominator);
    if (remapped < 0) {
        remapped = 0;
    }
    if (remapped > 255) {
        remapped = 255;
    }
    return remapped;
}

__host__ __device__ inline void insert_candidate(
    const double candidate_distance,
    const double candidate_x,
    const double candidate_y,
    const double candidate_z,
    const int candidate_index,
    const IntensityType candidate_intensity,
    const int k,
    const double* current_xs,
    const double* current_ys,
    const double* current_zs,
    double* best_distances,
    int* best_indices,
    IntensityType* best_intensities) {
    if (!better_candidate(
            candidate_distance,
            candidate_x,
            candidate_y,
            candidate_z,
            candidate_index,
            best_distances[k - 1],
            best_indices[k - 1],
            current_xs,
            current_ys,
            current_zs)) {
        return;
    }

    int insert_position = k - 1;
    while (insert_position > 0 &&
           better_candidate(
               candidate_distance,
               candidate_x,
               candidate_y,
               candidate_z,
               candidate_index,
               best_distances[insert_position - 1],
               best_indices[insert_position - 1],
               current_xs,
               current_ys,
               current_zs)) {
        best_distances[insert_position] = best_distances[insert_position - 1];
        best_indices[insert_position] = best_indices[insert_position - 1];
        best_intensities[insert_position] = best_intensities[insert_position - 1];
        --insert_position;
    }

    best_distances[insert_position] = candidate_distance;
    best_indices[insert_position] = candidate_index;
    best_intensities[insert_position] = candidate_intensity;
}

__host__ __device__ inline int compare_cell_triplets(
    const long long lhs_x,
    const long long lhs_y,
    const long long lhs_z,
    const long long rhs_x,
    const long long rhs_y,
    const long long rhs_z);

bool cell_point_less(const CellPoint& lhs, const CellPoint& rhs) {
    const int cmp = compare_cell_triplets(
        lhs.cell_x,
        lhs.cell_y,
        lhs.cell_z,
        rhs.cell_x,
        rhs.cell_y,
        rhs.cell_z);
    if (cmp != 0) {
        return cmp < 0;
    }
    return lhs.original_index < rhs.original_index;
}

__host__ __device__ inline int compare_cell_triplets(
    const long long lhs_x,
    const long long lhs_y,
    const long long lhs_z,
    const long long rhs_x,
    const long long rhs_y,
    const long long rhs_z) {
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

__device__ inline int find_cell_range(
    const long long* cell_xs,
    const long long* cell_ys,
    const long long* cell_zs,
    const int* cell_hash_values,
    const int cell_hash_mask,
    const int num_cells,
    const long long query_x,
    const long long query_y,
    const long long query_z) {
    if (cell_hash_mask == 0 || cell_hash_values == nullptr) {
        return -1;
    }

    uint64_t key = encode_cell_coords(query_x, query_y, query_z);
    int slot = static_cast<int>(key & static_cast<uint64_t>(cell_hash_mask));
    while (true) {
        const int cell_id = cell_hash_values[slot];
        if (cell_id < 0) {
            return -1;
        }
        if (cell_id < num_cells &&
            cell_xs[cell_id] == query_x &&
            cell_ys[cell_id] == query_y &&
            cell_zs[cell_id] == query_z) {
            return cell_id;
        }
        slot = (slot + 1) & cell_hash_mask;
    }
}

__host__ __device__ inline long long coordinate_to_cell(
    const double coordinate,
    const double inv_cell_size) {
    return static_cast<long long>(floor(coordinate * inv_cell_size));
}

__host__ __device__ inline double shell_outside_lower_bound_sq(
    const double x,
    const double y,
    const double z,
    const long long cell_x,
    const long long cell_y,
    const long long cell_z,
    const double cell_size,
    const int shell_radius) {
    const double left_x = x - static_cast<double>(cell_x - shell_radius) * cell_size;
    const double right_x = static_cast<double>(cell_x + shell_radius + 1) * cell_size - x;
    const double left_y = y - static_cast<double>(cell_y - shell_radius) * cell_size;
    const double right_y = static_cast<double>(cell_y + shell_radius + 1) * cell_size - y;
    const double left_z = z - static_cast<double>(cell_z - shell_radius) * cell_size;
    const double right_z = static_cast<double>(cell_z + shell_radius + 1) * cell_size - z;

    double min_axis_distance = left_x < right_x ? left_x : right_x;
    const double y_distance = left_y < right_y ? left_y : right_y;
    const double z_distance = left_z < right_z ? left_z : right_z;
    if (y_distance < min_axis_distance) {
        min_axis_distance = y_distance;
    }
    if (z_distance < min_axis_distance) {
        min_axis_distance = z_distance;
    }

    return min_axis_distance * min_axis_distance;
}

__host__ __device__ inline uint64_t encode_cell_coords(
    const long long x,
    const long long y,
    const long long z) {
    const uint64_t ux = static_cast<uint64_t>(x);
    const uint64_t uy = static_cast<uint64_t>(y);
    const uint64_t uz = static_cast<uint64_t>(z);
    return (ux * kCellHashMulX) ^ (uy * kCellHashMulY) ^ (uz * kCellHashMulZ);
}

void build_cell_hash_table(SpatialIndex& index) {
    const int num_cells = static_cast<int>(index.cell_xs.size());
    if (num_cells == 0) {
        index.cell_hash_mask = 0;
        index.cell_hash_values.clear();
        return;
    }

    int table_size = 1;
    while (table_size < num_cells * 2) {
        table_size <<= 1;
    }
    index.cell_hash_mask = table_size - 1;
    index.cell_hash_values.assign(table_size, -1);

    for (int cell_id = 0; cell_id < num_cells; ++cell_id) {
        const uint64_t key = encode_cell_coords(
            index.cell_xs[cell_id],
            index.cell_ys[cell_id],
            index.cell_zs[cell_id]);
        int slot = static_cast<int>(key & static_cast<uint64_t>(index.cell_hash_mask));
        while (index.cell_hash_values[slot] != -1) {
            slot = (slot + 1) & index.cell_hash_mask;
        }
        index.cell_hash_values[slot] = cell_id;
    }
}

__device__ inline void scan_shell(
    const long long query_cell_x,
    const long long query_cell_y,
    const long long query_cell_z,
    const double query_x,
    const double query_y,
    const double query_z,
    const int query_index,
    const int radius,
    const bool border_only,
    const double* sorted_xs,
    const double* sorted_ys,
    const double* sorted_zs,
    const IntensityType* sorted_intensities,
    const int* sorted_original_indices,
    const long long* cell_xs,
    const long long* cell_ys,
    const long long* cell_zs,
    const int* cell_hash_values,
    const int cell_hash_mask,
    const int* cell_starts,
    const int* cell_ends,
    const int num_cells,
    const int k,
    const double* original_xs,
    const double* original_ys,
    const double* original_zs,
    int* candidate_count,
    double* best_distances,
    int* best_indices,
    IntensityType* best_intensities) {
    for (int dx = -radius; dx <= radius; ++dx) {
        for (int dy = -radius; dy <= radius; ++dy) {
            for (int dz = -radius; dz <= radius; ++dz) {
                if (border_only) {
                    const int abs_dx = dx < 0 ? -dx : dx;
                    const int abs_dy = dy < 0 ? -dy : dy;
                    const int abs_dz = dz < 0 ? -dz : dz;
                    int max_abs = abs_dx;
                    if (abs_dy > max_abs) {
                        max_abs = abs_dy;
                    }
                    if (abs_dz > max_abs) {
                        max_abs = abs_dz;
                    }
                    if (max_abs != radius) {
                        continue;
                    }
                }

                const int cell_id = find_cell_range(
                    cell_xs,
                    cell_ys,
                    cell_zs,
                    cell_hash_values,
                    cell_hash_mask,
                    num_cells,
                    query_cell_x + dx,
                    query_cell_y + dy,
                    query_cell_z + dz);
                if (cell_id < 0) {
                    continue;
                }

                for (int position = cell_starts[cell_id]; position < cell_ends[cell_id]; ++position) {
                    const int candidate_index = sorted_original_indices[position];
                    if (candidate_index == query_index) {
                        continue;
                    }

                    ++(*candidate_count);
                    const double delta_x = query_x - sorted_xs[position];
                    const double delta_y = query_y - sorted_ys[position];
                    const double delta_z = query_z - sorted_zs[position];
                    const double squared_distance =
                        delta_x * delta_x + delta_y * delta_y + delta_z * delta_z;
                    insert_candidate(
                        squared_distance,
                        sorted_xs[position],
                        sorted_ys[position],
                        sorted_zs[position],
                        candidate_index,
                        sorted_intensities[position],
                        k,
                        original_xs,
                        original_ys,
                        original_zs,
                        best_distances,
                        best_indices,
                        best_intensities);
                }
            }
        }
    }
}

template <int BlockSize>
__global__ void approx_knn_equalize_kernel(
    const double* __restrict__ query_xs,
    const double* __restrict__ query_ys,
    const double* __restrict__ query_zs,
    const IntensityType* __restrict__ query_intensities,
    const double* __restrict__ sorted_xs,
    const double* __restrict__ sorted_ys,
    const double* __restrict__ sorted_zs,
    const IntensityType* __restrict__ sorted_intensities,
    const int* __restrict__ sorted_original_indices,
    const long long* __restrict__ cell_xs,
    const long long* __restrict__ cell_ys,
    const long long* __restrict__ cell_zs,
    const int* __restrict__ cell_starts,
    const int* __restrict__ cell_ends,
    const int* __restrict__ cell_hash_values,
    const int cell_hash_mask,
    const int num_cells,
    const int n,
    const int k,
    const double cell_size,
    const double inv_cell_size,
    int* __restrict__ output_intensities,
    FlagType* __restrict__ fallback_flags) {
    const int point_index = blockIdx.x * BlockSize + threadIdx.x;
    if (point_index >= n) {
        return;
    }

    double best_distances[kMaxK];
    int best_indices[kMaxK];
    IntensityType best_intensities[kMaxK];
    for (int neighbor = 0; neighbor < k; ++neighbor) {
        best_distances[neighbor] = 1.0 / 0.0;
        best_indices[neighbor] = -1;
        best_intensities[neighbor] = 0;
    }

    const double query_x = query_xs[point_index];
    const double query_y = query_ys[point_index];
    const double query_z = query_zs[point_index];
    const IntensityType center_intensity = query_intensities[point_index];
    const long long query_cell_x = coordinate_to_cell(query_x, inv_cell_size);
    const long long query_cell_y = coordinate_to_cell(query_y, inv_cell_size);
    const long long query_cell_z = coordinate_to_cell(query_z, inv_cell_size);

    int candidate_count = 0;
    bool resolved = false;

    scan_shell(
        query_cell_x,
        query_cell_y,
        query_cell_z,
        query_x,
        query_y,
        query_z,
        point_index,
        kFirstShellRadius,
        false,
        sorted_xs,
        sorted_ys,
        sorted_zs,
        sorted_intensities,
        sorted_original_indices,
        cell_xs,
        cell_ys,
        cell_zs,
        cell_hash_values,
        cell_hash_mask,
        cell_starts,
        cell_ends,
        num_cells,
        k,
        query_xs,
        query_ys,
        query_zs,
        &candidate_count,
        best_distances,
        best_indices,
        best_intensities);

    if (candidate_count >= k) {
        const double lower_bound_sq = shell_outside_lower_bound_sq(
            query_x,
            query_y,
            query_z,
            query_cell_x,
            query_cell_y,
            query_cell_z,
            cell_size,
            kFirstShellRadius);
        resolved = best_distances[k - 1] <= lower_bound_sq;
    }

    if (!resolved) {
        scan_shell(
            query_cell_x,
            query_cell_y,
            query_cell_z,
            query_x,
            query_y,
            query_z,
            point_index,
            kSecondShellRadius,
            true,
            sorted_xs,
            sorted_ys,
            sorted_zs,
            sorted_intensities,
            sorted_original_indices,
            cell_xs,
            cell_ys,
            cell_zs,
            cell_hash_values,
            cell_hash_mask,
            cell_starts,
            cell_ends,
            num_cells,
            k,
            query_xs,
            query_ys,
            query_zs,
            &candidate_count,
            best_distances,
            best_indices,
            best_intensities);

        if (candidate_count >= k) {
            const double lower_bound_sq = shell_outside_lower_bound_sq(
                query_x,
                query_y,
                query_z,
                query_cell_x,
                query_cell_y,
                query_cell_z,
                cell_size,
                kSecondShellRadius);
            resolved = best_distances[k - 1] <= lower_bound_sq;
        }
    }

    output_intensities[point_index] = remap_intensity_from_neighbors(
        center_intensity,
        best_intensities,
        k);
    fallback_flags[point_index] = resolved ? 0 : 1;
}

template <int BlockSize>
__global__ void exact_fallback_kernel(
    const double* __restrict__ xs,
    const double* __restrict__ ys,
    const double* __restrict__ zs,
    const IntensityType* __restrict__ intensities,
    const int n,
    const int k,
    const int fallback_count,
    const int* __restrict__ fallback_queries,
    int* __restrict__ output_intensities) {
    __shared__ double tile_xs[BlockSize];
    __shared__ double tile_ys[BlockSize];
    __shared__ double tile_zs[BlockSize];
    __shared__ IntensityType tile_intensities[BlockSize];

    const int fallback_id = blockIdx.x * BlockSize + threadIdx.x;
    const bool active = fallback_id < fallback_count;

    double best_distances[kMaxK];
    int best_indices[kMaxK];
    IntensityType best_intensities[kMaxK];

    int point_index = 0;
    double query_x = 0.0;
    double query_y = 0.0;
    double query_z = 0.0;
    IntensityType center_intensity = 0;

    if (active) {
        point_index = fallback_queries[fallback_id];
        query_x = xs[point_index];
        query_y = ys[point_index];
        query_z = zs[point_index];
        center_intensity = intensities[point_index];
        for (int neighbor = 0; neighbor < k; ++neighbor) {
            best_distances[neighbor] = 1.0 / 0.0;
            best_indices[neighbor] = -1;
            best_intensities[neighbor] = 0;
        }
    }

    for (int tile_start = 0; tile_start < n; tile_start += BlockSize) {
        const int load_index = tile_start + threadIdx.x;
        if (load_index < n) {
            tile_xs[threadIdx.x] = xs[load_index];
            tile_ys[threadIdx.x] = ys[load_index];
            tile_zs[threadIdx.x] = zs[load_index];
            tile_intensities[threadIdx.x] = intensities[load_index];
        }
        __syncthreads();

        if (active) {
            const int tile_count = (n - tile_start) < BlockSize ? (n - tile_start) : BlockSize;
            for (int tile_offset = 0; tile_offset < tile_count; ++tile_offset) {
                const int candidate_index = tile_start + tile_offset;
                if (candidate_index == point_index) {
                    continue;
                }

                const double delta_x = query_x - tile_xs[tile_offset];
                const double delta_y = query_y - tile_ys[tile_offset];
                const double delta_z = query_z - tile_zs[tile_offset];
                const double squared_distance =
                    delta_x * delta_x + delta_y * delta_y + delta_z * delta_z;
                insert_candidate(
                    squared_distance,
                    tile_xs[tile_offset],
                    tile_ys[tile_offset],
                    tile_zs[tile_offset],
                    candidate_index,
                    tile_intensities[tile_offset],
                    k,
                    xs,
                    ys,
                    zs,
                    best_distances,
                    best_indices,
                    best_intensities);
            }
        }
        __syncthreads();
    }

    if (!active) {
        return;
    }

    output_intensities[point_index] = remap_intensity_from_neighbors(
        center_intensity,
        best_intensities,
        k);
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
    if (data.k >= data.n) {
        throw std::runtime_error(
            "k must be strictly less than n because KNN excludes the center point itself from the neighbor set.");
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

        point.x = std::stod(point.x_text);
        point.y = std::stod(point.y_text);
        point.z = std::stod(point.z_text);
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

double sampled_kth_distance_sq(
    const HostArrays& arrays,
    const int n,
    const int k,
    const int point_index) {
    double best_distances[kMaxK];
    int best_indices[kMaxK];
    for (int neighbor = 0; neighbor < k; ++neighbor) {
        best_distances[neighbor] = std::numeric_limits<double>::infinity();
        best_indices[neighbor] = -1;
    }

    const double query_x = arrays.xs[point_index];
    const double query_y = arrays.ys[point_index];
    const double query_z = arrays.zs[point_index];

    for (int candidate = 0; candidate < n; ++candidate) {
        if (candidate == point_index) {
            continue;
        }

        const double delta_x = query_x - arrays.xs[candidate];
        const double delta_y = query_y - arrays.ys[candidate];
        const double delta_z = query_z - arrays.zs[candidate];
        const double squared_distance =
            delta_x * delta_x + delta_y * delta_y + delta_z * delta_z;

        if (!better_candidate(
                squared_distance,
                arrays.xs[candidate],
                arrays.ys[candidate],
                arrays.zs[candidate],
                candidate,
                best_distances[k - 1],
                best_indices[k - 1],
                arrays.xs.data(),
                arrays.ys.data(),
                arrays.zs.data())) {
            continue;
        }

        int insert_position = k - 1;
        while (insert_position > 0 &&
               better_candidate(
                   squared_distance,
                   arrays.xs[candidate],
                   arrays.ys[candidate],
                   arrays.zs[candidate],
                   candidate,
                   best_distances[insert_position - 1],
                   best_indices[insert_position - 1],
                   arrays.xs.data(),
                   arrays.ys.data(),
                   arrays.zs.data())) {
            best_distances[insert_position] = best_distances[insert_position - 1];
            best_indices[insert_position] = best_indices[insert_position - 1];
            --insert_position;
        }

        best_distances[insert_position] = squared_distance;
        best_indices[insert_position] = candidate;
    }

    return best_distances[k - 1];
}

double choose_cell_size(const InputData& data, const HostArrays& arrays) {
    const int sample_count = data.n < kSampleBudget ? data.n : kSampleBudget;
    std::vector<double> sampled_distances(sample_count, 1.0);

    #pragma omp parallel for schedule(static)
    for (int sample_id = 0; sample_id < sample_count; ++sample_id) {
        const int point_index = static_cast<int>(
            (static_cast<long long>(sample_id) * data.n) / sample_count);
        const double kth_distance_sq = sampled_kth_distance_sq(
            arrays,
            data.n,
            data.k,
            point_index);
        sampled_distances[sample_id] = std::sqrt(kth_distance_sq);
    }

    std::sort(sampled_distances.begin(), sampled_distances.end());
    int percentile_index = static_cast<int>(kVoxelDistancePercentile * sample_count);
    if (percentile_index >= sample_count) {
        percentile_index = sample_count - 1;
    }
    if (percentile_index < 0) {
        percentile_index = 0;
    }

    double cell_size = sampled_distances[percentile_index];
    if (!(cell_size > 0.0) || !std::isfinite(cell_size)) {
        cell_size = 1.0;
    }
    return cell_size;
}

SpatialIndex build_spatial_index(const InputData& data, const HostArrays& arrays) {
    SpatialIndex index;
    index.cell_size = choose_cell_size(data, arrays);
    index.inv_cell_size = 1.0 / index.cell_size;

    std::vector<CellPoint> entries(data.n);
    #pragma omp parallel for schedule(static)
    for (int point_index = 0; point_index < data.n; ++point_index) {
        entries[point_index].cell_x = coordinate_to_cell(
            arrays.xs[point_index],
            index.inv_cell_size);
        entries[point_index].cell_y = coordinate_to_cell(
            arrays.ys[point_index],
            index.inv_cell_size);
        entries[point_index].cell_z = coordinate_to_cell(
            arrays.zs[point_index],
            index.inv_cell_size);
        entries[point_index].original_index = point_index;
    }

    std::sort(entries.begin(), entries.end(), cell_point_less);

    index.sorted_xs.resize(data.n);
    index.sorted_ys.resize(data.n);
    index.sorted_zs.resize(data.n);
    index.sorted_intensities.resize(data.n);
    index.sorted_original_indices.resize(data.n);

    for (int sorted_index = 0; sorted_index < data.n; ++sorted_index) {
        const int original_index = entries[sorted_index].original_index;
        index.sorted_xs[sorted_index] = arrays.xs[original_index];
        index.sorted_ys[sorted_index] = arrays.ys[original_index];
        index.sorted_zs[sorted_index] = arrays.zs[original_index];
        index.sorted_intensities[sorted_index] = arrays.intensities[original_index];
        index.sorted_original_indices[sorted_index] = original_index;
    }

    int start = 0;
    while (start < data.n) {
        int end = start + 1;
        while (end < data.n &&
               entries[end].cell_x == entries[start].cell_x &&
               entries[end].cell_y == entries[start].cell_y &&
               entries[end].cell_z == entries[start].cell_z) {
            ++end;
        }

        index.cell_xs.push_back(entries[start].cell_x);
        index.cell_ys.push_back(entries[start].cell_y);
        index.cell_zs.push_back(entries[start].cell_z);
        index.cell_starts.push_back(start);
        index.cell_ends.push_back(end);
        start = end;
    }

    build_cell_hash_table(index);

    return index;
}

ApproxResult compute_approx_knn_gpu(
    const InputData& data,
    const HostArrays& arrays,
    const SpatialIndex& index) {
    double* d_query_xs = nullptr;
    double* d_query_ys = nullptr;
    double* d_query_zs = nullptr;
    IntensityType* d_query_intensities = nullptr;
    double* d_sorted_xs = nullptr;
    double* d_sorted_ys = nullptr;
    double* d_sorted_zs = nullptr;
    IntensityType* d_sorted_intensities = nullptr;
    int* d_sorted_original_indices = nullptr;
    long long* d_cell_xs = nullptr;
    long long* d_cell_ys = nullptr;
    long long* d_cell_zs = nullptr;
    int* d_cell_starts = nullptr;
    int* d_cell_ends = nullptr;
    int* d_cell_hash_values = nullptr;
    int* d_output_intensities = nullptr;
    FlagType* d_fallback_flags = nullptr;
    int* d_fallback_queries = nullptr;

    const std::size_t point_bytes = static_cast<std::size_t>(data.n) * sizeof(double);
    const std::size_t query_intensity_bytes =
        static_cast<std::size_t>(data.n) * sizeof(IntensityType);
    const std::size_t sorted_index_bytes =
        static_cast<std::size_t>(data.n) * sizeof(int);
    const std::size_t output_bytes = static_cast<std::size_t>(data.n) * sizeof(int);
    const std::size_t fallback_flag_bytes =
        static_cast<std::size_t>(data.n) * sizeof(FlagType);
    const int num_cells = static_cast<int>(index.cell_xs.size());
    const std::size_t cell_coord_bytes = static_cast<std::size_t>(num_cells) * sizeof(long long);
    const std::size_t cell_range_bytes = static_cast<std::size_t>(num_cells) * sizeof(int);
    const std::size_t hash_bytes = static_cast<std::size_t>(index.cell_hash_values.size()) * sizeof(int);

    try {
        CUDA_CHECK(cudaMalloc(&d_query_xs, point_bytes));
        CUDA_CHECK(cudaMalloc(&d_query_ys, point_bytes));
        CUDA_CHECK(cudaMalloc(&d_query_zs, point_bytes));
        CUDA_CHECK(cudaMalloc(&d_query_intensities, query_intensity_bytes));
        CUDA_CHECK(cudaMalloc(&d_sorted_xs, point_bytes));
        CUDA_CHECK(cudaMalloc(&d_sorted_ys, point_bytes));
        CUDA_CHECK(cudaMalloc(&d_sorted_zs, point_bytes));
        CUDA_CHECK(cudaMalloc(&d_sorted_intensities, query_intensity_bytes));
        CUDA_CHECK(cudaMalloc(&d_sorted_original_indices, sorted_index_bytes));
        CUDA_CHECK(cudaMalloc(&d_cell_xs, cell_coord_bytes));
        CUDA_CHECK(cudaMalloc(&d_cell_ys, cell_coord_bytes));
        CUDA_CHECK(cudaMalloc(&d_cell_zs, cell_coord_bytes));
        CUDA_CHECK(cudaMalloc(&d_cell_starts, cell_range_bytes));
        CUDA_CHECK(cudaMalloc(&d_cell_ends, cell_range_bytes));
        if (!index.cell_hash_values.empty()) {
            CUDA_CHECK(cudaMalloc(&d_cell_hash_values, hash_bytes));
        }
        CUDA_CHECK(cudaMalloc(&d_output_intensities, output_bytes));
        CUDA_CHECK(cudaMalloc(&d_fallback_flags, fallback_flag_bytes));

        CUDA_CHECK(cudaMemcpy(
            d_query_xs,
            arrays.xs.data(),
            point_bytes,
            cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(
            d_query_ys,
            arrays.ys.data(),
            point_bytes,
            cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(
            d_query_zs,
            arrays.zs.data(),
            point_bytes,
            cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(
            d_query_intensities,
            arrays.intensities.data(),
            query_intensity_bytes,
            cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(
            d_sorted_xs,
            index.sorted_xs.data(),
            point_bytes,
            cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(
            d_sorted_ys,
            index.sorted_ys.data(),
            point_bytes,
            cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(
            d_sorted_zs,
            index.sorted_zs.data(),
            point_bytes,
            cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(
            d_sorted_intensities,
            index.sorted_intensities.data(),
            query_intensity_bytes,
            cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(
            d_sorted_original_indices,
            index.sorted_original_indices.data(),
            sorted_index_bytes,
            cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(
            d_cell_xs,
            index.cell_xs.data(),
            cell_coord_bytes,
            cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(
            d_cell_ys,
            index.cell_ys.data(),
            cell_coord_bytes,
            cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(
            d_cell_zs,
            index.cell_zs.data(),
            cell_coord_bytes,
            cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(
            d_cell_starts,
            index.cell_starts.data(),
            cell_range_bytes,
            cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(
            d_cell_ends,
            index.cell_ends.data(),
            cell_range_bytes,
            cudaMemcpyHostToDevice));
        if (!index.cell_hash_values.empty()) {
            CUDA_CHECK(cudaMemcpy(
                d_cell_hash_values,
                index.cell_hash_values.data(),
                hash_bytes,
                cudaMemcpyHostToDevice));
        }

        const int blocks = (data.n + kThreadsPerBlock - 1) / kThreadsPerBlock;
        approx_knn_equalize_kernel<kThreadsPerBlock><<<blocks, kThreadsPerBlock>>>(
            d_query_xs,
            d_query_ys,
            d_query_zs,
            d_query_intensities,
            d_sorted_xs,
            d_sorted_ys,
            d_sorted_zs,
            d_sorted_intensities,
            d_sorted_original_indices,
            d_cell_xs,
            d_cell_ys,
            d_cell_zs,
            d_cell_starts,
            d_cell_ends,
            d_cell_hash_values,
            index.cell_hash_mask,
            num_cells,
            data.n,
            data.k,
            index.cell_size,
            index.inv_cell_size,
            d_output_intensities,
            d_fallback_flags);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<FlagType> fallback_flags(data.n, 0);
        CUDA_CHECK(cudaMemcpy(
            fallback_flags.data(),
            d_fallback_flags,
            fallback_flag_bytes,
            cudaMemcpyDeviceToHost));

        std::vector<int> fallback_queries;
        fallback_queries.reserve(data.n / 8 + 1);
        for (int point_index = 0; point_index < data.n; ++point_index) {
            if (fallback_flags[point_index] != 0) {
                fallback_queries.push_back(point_index);
            }
        }

        if (!fallback_queries.empty()) {
            const std::size_t fallback_query_bytes =
                static_cast<std::size_t>(fallback_queries.size()) * sizeof(int);
            CUDA_CHECK(cudaMalloc(&d_fallback_queries, fallback_query_bytes));
            CUDA_CHECK(cudaMemcpy(
                d_fallback_queries,
                fallback_queries.data(),
                fallback_query_bytes,
                cudaMemcpyHostToDevice));

            const int fallback_blocks =
                (static_cast<int>(fallback_queries.size()) + kThreadsPerBlock - 1) /
                kThreadsPerBlock;
            exact_fallback_kernel<kThreadsPerBlock><<<fallback_blocks, kThreadsPerBlock>>>(
                d_query_xs,
                d_query_ys,
                d_query_zs,
                d_query_intensities,
                data.n,
                data.k,
                static_cast<int>(fallback_queries.size()),
                d_fallback_queries,
                d_output_intensities);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
        }

        ApproxResult result;
        result.output_intensities.resize(data.n);
        result.cell_size = index.cell_size;
        result.num_cells = num_cells;
        result.fallback_count = static_cast<int>(fallback_queries.size());
        CUDA_CHECK(cudaMemcpy(
            result.output_intensities.data(),
            d_output_intensities,
            output_bytes,
            cudaMemcpyDeviceToHost));

        CUDA_CHECK(cudaFree(d_query_xs));
        CUDA_CHECK(cudaFree(d_query_ys));
        CUDA_CHECK(cudaFree(d_query_zs));
        CUDA_CHECK(cudaFree(d_query_intensities));
        CUDA_CHECK(cudaFree(d_sorted_xs));
        CUDA_CHECK(cudaFree(d_sorted_ys));
        CUDA_CHECK(cudaFree(d_sorted_zs));
        CUDA_CHECK(cudaFree(d_sorted_intensities));
        CUDA_CHECK(cudaFree(d_sorted_original_indices));
        CUDA_CHECK(cudaFree(d_cell_xs));
        CUDA_CHECK(cudaFree(d_cell_ys));
        CUDA_CHECK(cudaFree(d_cell_zs));
        CUDA_CHECK(cudaFree(d_cell_starts));
        CUDA_CHECK(cudaFree(d_cell_ends));
        CUDA_CHECK(cudaFree(d_cell_hash_values));
        CUDA_CHECK(cudaFree(d_output_intensities));
        CUDA_CHECK(cudaFree(d_fallback_flags));
        CUDA_CHECK(cudaFree(d_fallback_queries));

        return result;
    } catch (...) {
        cudaFree(d_query_xs);
        cudaFree(d_query_ys);
        cudaFree(d_query_zs);
        cudaFree(d_query_intensities);
        cudaFree(d_sorted_xs);
        cudaFree(d_sorted_ys);
        cudaFree(d_sorted_zs);
        cudaFree(d_sorted_intensities);
        cudaFree(d_sorted_original_indices);
        cudaFree(d_cell_xs);
        cudaFree(d_cell_ys);
        cudaFree(d_cell_zs);
        cudaFree(d_cell_starts);
        cudaFree(d_cell_ends);
        cudaFree(d_cell_hash_values);
        cudaFree(d_output_intensities);
        cudaFree(d_fallback_flags);
        cudaFree(d_fallback_queries);
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

void print_usage(const char* program_name) {
    std::cerr << "Usage: " << program_name
              << " [--stats] <input_file> [output_file]\n";
}

}  // namespace

int main(int argc, char** argv) {
    try {
        bool print_stats = false;
        std::vector<std::string> positional;

        for (int arg_index = 1; arg_index < argc; ++arg_index) {
            const std::string argument = argv[arg_index];
            if (argument == "--stats") {
                print_stats = true;
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
            positional.size() == 2 ? positional[1] : "approx_knn.txt";

        const InputData data = read_input(input_path);
        const HostArrays arrays = build_host_arrays(data);
        const SpatialIndex index = build_spatial_index(data, arrays);
        const ApproxResult result = compute_approx_knn_gpu(data, arrays, index);

        write_output(output_path, data, result.output_intensities);

        if (print_stats) {
            std::cerr << "cell_size=" << result.cell_size
                      << " cells=" << result.num_cells
                      << " fallback_points=" << result.fallback_count << '\n';
        }

        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
