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
constexpr double kVoxelDistancePercentile = 0.60;
using IntensityType = std::uint8_t;
using HistogramCountType = int;
using DistanceType = long long;
constexpr DistanceType kInfiniteDistance = 0x7fffffffffffffffLL;
constexpr uint64_t kCellHashMulX = 0x9e3779b97f4a7c15ULL;
constexpr uint64_t kCellHashMulY = 0xc2b2ae3d27d4eb4fULL;
constexpr uint64_t kCellHashMulZ = 0x165667b19e3779f9ULL;

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
    std::vector<int> sorted_xs;
    std::vector<int> sorted_ys;
    std::vector<int> sorted_zs;
    std::vector<IntensityType> sorted_intensities;
    std::vector<int> sorted_original_indices;
    std::vector<int> cell_hash_values;
    int cell_hash_mask = 0;
};

struct DeviceSpatialIndex {
    int n = 0;
    int num_cells = 0;
    double cell_size = 1.0;
    double inv_cell_size = 1.0;
    int* sorted_xs = nullptr;
    int* sorted_ys = nullptr;
    int* sorted_zs = nullptr;
    IntensityType* sorted_intensities = nullptr;
    int* sorted_original_indices = nullptr;
    long long* cell_xs = nullptr;
    long long* cell_ys = nullptr;
    long long* cell_zs = nullptr;
    int* cell_starts = nullptr;
    int* cell_ends = nullptr;
    int* cell_hash_values = nullptr;
    int cell_hash_mask = 0;
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
        oss << "Point " << index << " has a non-integer " << axis_name << " coordinate.";
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

__host__ __device__ inline bool better_candidate(
    const DistanceType candidate_distance,
    const int candidate_x,
    const int candidate_y,
    const int candidate_z,
    const int candidate_index,
    const DistanceType current_distance,
    const int current_index,
    const int* current_xs,
    const int* current_ys,
    const int* current_zs) {
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

__host__ __device__ inline DistanceType squared_distance(
    const int lhs_x,
    const int lhs_y,
    const int lhs_z,
    const int rhs_x,
    const int rhs_y,
    const int rhs_z) {
    const DistanceType dx = static_cast<DistanceType>(lhs_x) - static_cast<DistanceType>(rhs_x);
    const DistanceType dy = static_cast<DistanceType>(lhs_y) - static_cast<DistanceType>(rhs_y);
    const DistanceType dz = static_cast<DistanceType>(lhs_z) - static_cast<DistanceType>(rhs_z);
    return dx * dx + dy * dy + dz * dz;
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

__host__ __device__ inline void insert_candidate(
    const DistanceType candidate_distance,
    const int candidate_x,
    const int candidate_y,
    const int candidate_z,
    const int candidate_index,
    const IntensityType candidate_intensity,
    const int k,
    const int* current_xs,
    const int* current_ys,
    const int* current_zs,
    DistanceType* best_distances,
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

InputData read_input(const std::string& path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("Failed to open input file: " + path);
    }

    InputData data;
    if (!(input >> data.n >> data.k >> data.t)) {
        throw std::runtime_error("Failed to read n, k, and T from input file.");
    }

    if (data.n <= 0 || data.n > kMaxN) {
        throw std::runtime_error("n must be in [1, 100000].");
    }
    if (data.k <= 0 || data.k > kMaxK) {
        throw std::runtime_error("k must be in [1, 128].");
    }
    if (data.k >= data.n) {
        throw std::runtime_error("k must be strictly less than n because KNN excludes the center point itself from the neighbor set.");
    }
    if (data.t <= 0 || data.t > kMaxT) {
        throw std::runtime_error("T must be in [1, 50].");
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

std::vector<int> compute_exact_knn_cpu(const InputData& data, const HostArrays& arrays) {
    std::vector<int> output(data.n, 0);

    #pragma omp parallel for schedule(static)
    for (int point_index = 0; point_index < data.n; ++point_index) {
        DistanceType best_distances[kMaxK];
        int best_indices[kMaxK];
        IntensityType best_intensities[kMaxK];
        for (int neighbor = 0; neighbor < data.k; ++neighbor) {
            best_distances[neighbor] = kInfiniteDistance;
            best_indices[neighbor] = -1;
            best_intensities[neighbor] = 0;
        }

        const int query_x = arrays.xs[point_index];
        const int query_y = arrays.ys[point_index];
        const int query_z = arrays.zs[point_index];
        for (int candidate = 0; candidate < data.n; ++candidate) {
            if (candidate == point_index) {
                continue;
            }

            const DistanceType distance = squared_distance(
                query_x,
                query_y,
                query_z,
                arrays.xs[candidate],
                arrays.ys[candidate],
                arrays.zs[candidate]);
            insert_candidate(
                distance,
                arrays.xs[candidate],
                arrays.ys[candidate],
                arrays.zs[candidate],
                candidate,
                arrays.intensities[candidate],
                data.k,
                arrays.xs.data(),
                arrays.ys.data(),
                arrays.zs.data(),
                best_distances,
                best_indices,
                best_intensities);
        }

        HistogramCountType histogram[kIntensityLevels] = {};
        ++histogram[arrays.intensities[point_index]];
        for (int neighbor = 0; neighbor < data.k; ++neighbor) {
            ++histogram[best_intensities[neighbor]];
        }
        output[point_index] = remap_intensity(
            histogram,
            static_cast<int>(arrays.intensities[point_index]),
            data.k + 1);
    }

    return output;
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

__host__ __device__ inline long long coordinate_to_cell(
    const int coordinate,
    const double inv_cell_size) {
    return static_cast<long long>(floor(static_cast<double>(coordinate) * inv_cell_size));
}

__host__ __device__ inline double shell_outside_lower_bound_sq(
    const int x,
    const int y,
    const int z,
    const long long cell_x,
    const long long cell_y,
    const long long cell_z,
    const double cell_size,
    const int shell_radius) {
    const double left_x = static_cast<double>(x) - static_cast<double>(cell_x - shell_radius) * cell_size;
    const double right_x = static_cast<double>(cell_x + shell_radius + 1) * cell_size - static_cast<double>(x);
    const double left_y = static_cast<double>(y) - static_cast<double>(cell_y - shell_radius) * cell_size;
    const double right_y = static_cast<double>(cell_y + shell_radius + 1) * cell_size - static_cast<double>(y);
    const double left_z = static_cast<double>(z) - static_cast<double>(cell_z - shell_radius) * cell_size;
    const double right_z = static_cast<double>(cell_z + shell_radius + 1) * cell_size - static_cast<double>(z);

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

__device__ inline void scan_shell(
    const long long query_cell_x,
    const long long query_cell_y,
    const long long query_cell_z,
    const int query_x,
    const int query_y,
    const int query_z,
    const int query_index,
    const int radius,
    const bool border_only,
    const int* sorted_xs,
    const int* sorted_ys,
    const int* sorted_zs,
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
    const int* original_xs,
    const int* original_ys,
    const int* original_zs,
    int* candidate_count,
    DistanceType* best_distances,
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
                    const DistanceType distance = squared_distance(
                        query_x,
                        query_y,
                        query_z,
                        sorted_xs[position],
                        sorted_ys[position],
                        sorted_zs[position]);
                    insert_candidate(
                        distance,
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

double sampled_kth_distance_sq(
    const HostArrays& arrays,
    const int n,
    const int k,
    const int point_index) {
    DistanceType best_distances[kMaxK];
    int best_indices[kMaxK];
    for (int neighbor = 0; neighbor < k; ++neighbor) {
        best_distances[neighbor] = kInfiniteDistance;
        best_indices[neighbor] = -1;
    }

    const int query_x = arrays.xs[point_index];
    const int query_y = arrays.ys[point_index];
    const int query_z = arrays.zs[point_index];
    for (int candidate = 0; candidate < n; ++candidate) {
        if (candidate == point_index) {
            continue;
        }
        const DistanceType distance = squared_distance(
            query_x,
            query_y,
            query_z,
            arrays.xs[candidate],
            arrays.ys[candidate],
            arrays.zs[candidate]);
        if (!better_candidate(
                distance,
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
                   distance,
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
        best_distances[insert_position] = distance;
        best_indices[insert_position] = candidate;
    }
    return static_cast<double>(best_distances[k - 1]);
}

double choose_cell_size(const InputData& data, const HostArrays& arrays) {
    const int sample_count = data.n < kSampleBudget ? data.n : kSampleBudget;
    std::vector<double> sampled_distances(sample_count, 1.0);
    #pragma omp parallel for schedule(static)
    for (int sample_id = 0; sample_id < sample_count; ++sample_id) {
        const int point_index = static_cast<int>((static_cast<long long>(sample_id) * data.n) / sample_count);
        sampled_distances[sample_id] = std::sqrt(sampled_kth_distance_sq(arrays, data.n, data.k, point_index));
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
        entries[point_index].cell_x = coordinate_to_cell(arrays.xs[point_index], index.inv_cell_size);
        entries[point_index].cell_y = coordinate_to_cell(arrays.ys[point_index], index.inv_cell_size);
        entries[point_index].cell_z = coordinate_to_cell(arrays.zs[point_index], index.inv_cell_size);
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

DeviceSpatialIndex upload_spatial_index(const InputData& data, const SpatialIndex& index) {
    DeviceSpatialIndex device_index;
    device_index.n = data.n;
    device_index.num_cells = static_cast<int>(index.cell_xs.size());
    device_index.cell_size = index.cell_size;
    device_index.inv_cell_size = index.inv_cell_size;

    const std::size_t point_bytes = static_cast<std::size_t>(data.n) * sizeof(int);
    const std::size_t intensity_bytes = static_cast<std::size_t>(data.n) * sizeof(IntensityType);
    const std::size_t index_bytes = static_cast<std::size_t>(data.n) * sizeof(int);
    const std::size_t cell_coord_bytes = static_cast<std::size_t>(device_index.num_cells) * sizeof(long long);
    const std::size_t cell_range_bytes = static_cast<std::size_t>(device_index.num_cells) * sizeof(int);
    const std::size_t hash_bytes = static_cast<std::size_t>(index.cell_hash_values.size()) * sizeof(int);

    try {
        CUDA_CHECK(cudaMalloc(&device_index.sorted_xs, point_bytes));
        CUDA_CHECK(cudaMalloc(&device_index.sorted_ys, point_bytes));
        CUDA_CHECK(cudaMalloc(&device_index.sorted_zs, point_bytes));
        CUDA_CHECK(cudaMalloc(&device_index.sorted_intensities, intensity_bytes));
        CUDA_CHECK(cudaMalloc(&device_index.sorted_original_indices, index_bytes));
        CUDA_CHECK(cudaMalloc(&device_index.cell_xs, cell_coord_bytes));
        CUDA_CHECK(cudaMalloc(&device_index.cell_ys, cell_coord_bytes));
        CUDA_CHECK(cudaMalloc(&device_index.cell_zs, cell_coord_bytes));
        CUDA_CHECK(cudaMalloc(&device_index.cell_starts, cell_range_bytes));
        CUDA_CHECK(cudaMalloc(&device_index.cell_ends, cell_range_bytes));
        if (!index.cell_hash_values.empty()) {
            CUDA_CHECK(cudaMalloc(&device_index.cell_hash_values, hash_bytes));
        }

        CUDA_CHECK(cudaMemcpy(device_index.sorted_xs, index.sorted_xs.data(), point_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(device_index.sorted_ys, index.sorted_ys.data(), point_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(device_index.sorted_zs, index.sorted_zs.data(), point_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(device_index.sorted_intensities, index.sorted_intensities.data(), intensity_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(device_index.sorted_original_indices, index.sorted_original_indices.data(), index_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(device_index.cell_xs, index.cell_xs.data(), cell_coord_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(device_index.cell_ys, index.cell_ys.data(), cell_coord_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(device_index.cell_zs, index.cell_zs.data(), cell_coord_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(device_index.cell_starts, index.cell_starts.data(), cell_range_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(device_index.cell_ends, index.cell_ends.data(), cell_range_bytes, cudaMemcpyHostToDevice));
        if (!index.cell_hash_values.empty()) {
            CUDA_CHECK(cudaMemcpy(device_index.cell_hash_values, index.cell_hash_values.data(), hash_bytes, cudaMemcpyHostToDevice));
        }
    } catch (...) {
        cudaFree(device_index.sorted_xs);
        cudaFree(device_index.sorted_ys);
        cudaFree(device_index.sorted_zs);
        cudaFree(device_index.sorted_intensities);
        cudaFree(device_index.sorted_original_indices);
        cudaFree(device_index.cell_xs);
        cudaFree(device_index.cell_ys);
        cudaFree(device_index.cell_zs);
        cudaFree(device_index.cell_starts);
        cudaFree(device_index.cell_ends);
        cudaFree(device_index.cell_hash_values);
        throw;
    }
    device_index.cell_hash_mask = index.cell_hash_mask;
    return device_index;
}

void free_spatial_index(DeviceSpatialIndex& device_index) {
    cudaFree(device_index.sorted_xs);
    cudaFree(device_index.sorted_ys);
    cudaFree(device_index.sorted_zs);
    cudaFree(device_index.sorted_intensities);
    cudaFree(device_index.sorted_original_indices);
    cudaFree(device_index.cell_xs);
    cudaFree(device_index.cell_ys);
    cudaFree(device_index.cell_zs);
    cudaFree(device_index.cell_starts);
    cudaFree(device_index.cell_ends);
    cudaFree(device_index.cell_hash_values);
    device_index = DeviceSpatialIndex{};
}

template <int BlockSize>
__global__ void exact_grid_knn_equalize_kernel(
    const int* __restrict__ query_xs,
    const int* __restrict__ query_ys,
    const int* __restrict__ query_zs,
    const IntensityType* __restrict__ query_intensities,
    const int* __restrict__ sorted_xs,
    const int* __restrict__ sorted_ys,
    const int* __restrict__ sorted_zs,
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
    const long long min_cell_x,
    const long long max_cell_x,
    const long long min_cell_y,
    const long long max_cell_y,
    const long long min_cell_z,
    const long long max_cell_z,
    int* __restrict__ output_intensities) {
    const int point_index = blockIdx.x * BlockSize + threadIdx.x;
    if (point_index >= n) {
        return;
    }

    DistanceType best_distances[kMaxK];
    int best_indices[kMaxK];
    IntensityType best_intensities[kMaxK];
    for (int neighbor = 0; neighbor < k; ++neighbor) {
        best_distances[neighbor] = kInfiniteDistance;
        best_indices[neighbor] = -1;
        best_intensities[neighbor] = 0;
    }

    const int query_x = query_xs[point_index];
    const int query_y = query_ys[point_index];
    const int query_z = query_zs[point_index];
    const IntensityType center_intensity = query_intensities[point_index];
    const long long query_cell_x = coordinate_to_cell(query_x, inv_cell_size);
    const long long query_cell_y = coordinate_to_cell(query_y, inv_cell_size);
    const long long query_cell_z = coordinate_to_cell(query_z, inv_cell_size);

    int candidate_count = 0;
    scan_shell(
        query_cell_x,
        query_cell_y,
        query_cell_z,
        query_x,
        query_y,
        query_z,
        point_index,
        0,
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

    const long long dx_low = query_cell_x - min_cell_x;
    const long long dx_high = max_cell_x - query_cell_x;
    const long long dy_low = query_cell_y - min_cell_y;
    const long long dy_high = max_cell_y - query_cell_y;
    const long long dz_low = query_cell_z - min_cell_z;
    const long long dz_high = max_cell_z - query_cell_z;
    long long max_shell = dx_low;
    if (dx_high > max_shell) {
        max_shell = dx_high;
    }
    if (dy_low > max_shell) {
        max_shell = dy_low;
    }
    if (dy_high > max_shell) {
        max_shell = dy_high;
    }
    if (dz_low > max_shell) {
        max_shell = dz_low;
    }
    if (dz_high > max_shell) {
        max_shell = dz_high;
    }

    bool resolved = false;
    if (candidate_count >= k) {
        const double lower_bound_sq = shell_outside_lower_bound_sq(
            query_x,
            query_y,
            query_z,
            query_cell_x,
            query_cell_y,
            query_cell_z,
            cell_size,
            0);
        resolved = static_cast<double>(best_distances[k - 1]) <= lower_bound_sq;
    }

    for (int radius = 1; !resolved && static_cast<long long>(radius) <= max_shell; ++radius) {
        scan_shell(
            query_cell_x,
            query_cell_y,
            query_cell_z,
            query_x,
            query_y,
            query_z,
            point_index,
            radius,
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

        if (candidate_count < k) {
            continue;
        }
        if (static_cast<long long>(radius) == max_shell) {
            resolved = true;
            break;
        }
        const double lower_bound_sq = shell_outside_lower_bound_sq(
            query_x,
            query_y,
            query_z,
            query_cell_x,
            query_cell_y,
            query_cell_z,
            cell_size,
            radius);
        resolved = static_cast<double>(best_distances[k - 1]) <= lower_bound_sq;
    }

    HistogramCountType histogram[kIntensityLevels];
    for (int value = 0; value < kIntensityLevels; ++value) {
        histogram[value] = 0;
    }
    ++histogram[center_intensity];
    for (int neighbor = 0; neighbor < k; ++neighbor) {
        ++histogram[best_intensities[neighbor]];
    }
    output_intensities[point_index] =
        remap_intensity(histogram, static_cast<int>(center_intensity), k + 1);
}

std::vector<int> compute_exact_knn_gpu(const InputData& data, const HostArrays& arrays) {
    int* d_xs = nullptr;
    int* d_ys = nullptr;
    int* d_zs = nullptr;
    IntensityType* d_intensities = nullptr;
    int* d_output_intensities = nullptr;

    const std::size_t point_bytes = static_cast<std::size_t>(data.n) * sizeof(int);
    const std::size_t intensity_bytes = static_cast<std::size_t>(data.n) * sizeof(IntensityType);
    const std::size_t output_bytes = static_cast<std::size_t>(data.n) * sizeof(int);

    SpatialIndex spatial_index = build_spatial_index(data, arrays);
    DeviceSpatialIndex device_index = upload_spatial_index(data, spatial_index);

    long long min_cell_x = 0;
    long long max_cell_x = 0;
    long long min_cell_y = 0;
    long long max_cell_y = 0;
    long long min_cell_z = 0;
    long long max_cell_z = 0;
    if (!spatial_index.cell_xs.empty()) {
        min_cell_x = max_cell_x = spatial_index.cell_xs[0];
        min_cell_y = max_cell_y = spatial_index.cell_ys[0];
        min_cell_z = max_cell_z = spatial_index.cell_zs[0];
        for (std::size_t cell = 1; cell < spatial_index.cell_xs.size(); ++cell) {
            if (spatial_index.cell_xs[cell] < min_cell_x) {
                min_cell_x = spatial_index.cell_xs[cell];
            }
            if (spatial_index.cell_xs[cell] > max_cell_x) {
                max_cell_x = spatial_index.cell_xs[cell];
            }
            if (spatial_index.cell_ys[cell] < min_cell_y) {
                min_cell_y = spatial_index.cell_ys[cell];
            }
            if (spatial_index.cell_ys[cell] > max_cell_y) {
                max_cell_y = spatial_index.cell_ys[cell];
            }
            if (spatial_index.cell_zs[cell] < min_cell_z) {
                min_cell_z = spatial_index.cell_zs[cell];
            }
            if (spatial_index.cell_zs[cell] > max_cell_z) {
                max_cell_z = spatial_index.cell_zs[cell];
            }
        }
    }

    try {
        CUDA_CHECK(cudaMalloc(&d_xs, point_bytes));
        CUDA_CHECK(cudaMalloc(&d_ys, point_bytes));
        CUDA_CHECK(cudaMalloc(&d_zs, point_bytes));
        CUDA_CHECK(cudaMalloc(&d_intensities, intensity_bytes));
        CUDA_CHECK(cudaMalloc(&d_output_intensities, output_bytes));

        CUDA_CHECK(cudaMemcpy(d_xs, arrays.xs.data(), point_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_ys, arrays.ys.data(), point_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_zs, arrays.zs.data(), point_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_intensities, arrays.intensities.data(), intensity_bytes, cudaMemcpyHostToDevice));

        const int blocks = (data.n + kThreadsPerBlock - 1) / kThreadsPerBlock;
        exact_grid_knn_equalize_kernel<kThreadsPerBlock><<<blocks, kThreadsPerBlock>>>(
            d_xs,
            d_ys,
            d_zs,
            d_intensities,
            device_index.sorted_xs,
            device_index.sorted_ys,
            device_index.sorted_zs,
        device_index.sorted_intensities,
        device_index.sorted_original_indices,
        device_index.cell_xs,
        device_index.cell_ys,
        device_index.cell_zs,
        device_index.cell_starts,
        device_index.cell_ends,
        device_index.cell_hash_values,
        device_index.cell_hash_mask,
        device_index.num_cells,
            data.n,
            data.k,
            device_index.cell_size,
            device_index.inv_cell_size,
            min_cell_x,
            max_cell_x,
            min_cell_y,
            max_cell_y,
            min_cell_z,
            max_cell_z,
            d_output_intensities);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<int> output(data.n, 0);
        CUDA_CHECK(cudaMemcpy(output.data(), d_output_intensities, output_bytes, cudaMemcpyDeviceToHost));

        CUDA_CHECK(cudaFree(d_xs));
        CUDA_CHECK(cudaFree(d_ys));
        CUDA_CHECK(cudaFree(d_zs));
        CUDA_CHECK(cudaFree(d_intensities));
        CUDA_CHECK(cudaFree(d_output_intensities));
        free_spatial_index(device_index);
        return output;
    } catch (...) {
        cudaFree(d_xs);
        cudaFree(d_ys);
        cudaFree(d_zs);
        cudaFree(d_intensities);
        cudaFree(d_output_intensities);
        free_spatial_index(device_index);
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
    const std::vector<int> cpu_output = compute_exact_knn_cpu(data, arrays);

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
        const std::string output_path = positional.size() == 2 ? positional[1] : "knn.txt";

        const InputData data = read_input(input_path);
        const HostArrays arrays = build_host_arrays(data);

        const std::vector<int> output_intensities = cpu_reference
            ? compute_exact_knn_cpu(data, arrays)
            : compute_exact_knn_gpu(data, arrays);

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
