#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <omp.h>

namespace {

#ifndef A2_APPROX_USE_SECOND_SHELL
#define A2_APPROX_USE_SECOND_SHELL 1
#endif

#ifndef A2_APPROX_USE_THIRD_SHELL
#define A2_APPROX_USE_THIRD_SHELL 1
#endif

#ifndef A2_APPROX_THIRD_SHELL_RADIUS
#define A2_APPROX_THIRD_SHELL_RADIUS 2
#endif

#ifndef A2_APPROX_OWN_CELL_ACCEPT_NUM
#define A2_APPROX_OWN_CELL_ACCEPT_NUM 5
#endif

#ifndef A2_APPROX_OWN_CELL_ACCEPT_DEN
#define A2_APPROX_OWN_CELL_ACCEPT_DEN 4
#endif

#ifndef A2_APPROX_THIRD_SHELL_TRIGGER_NUM
#define A2_APPROX_THIRD_SHELL_TRIGGER_NUM 5
#endif

#ifndef A2_APPROX_THIRD_SHELL_TRIGGER_DEN
#define A2_APPROX_THIRD_SHELL_TRIGGER_DEN 4
#endif

constexpr int kIntensityLevels = 256;
constexpr int kMaxK = 128;
constexpr int kMaxN = 100000;
constexpr int kMaxT = 50;
constexpr int kExactThreadsPerBlock = 128;
constexpr int kApproxThreadsPerBlock = 128;
constexpr int kKMeansThreadsPerBlock = 256;
constexpr int kKMeansHistogramHashSize = 1024;
constexpr int kSampleBudget = 1024;
constexpr int kFirstShellRadius = 0;
constexpr int kSecondShellRadius = 1;
constexpr bool kApproxUseSecondShell = A2_APPROX_USE_SECOND_SHELL != 0;
constexpr bool kApproxUseThirdShell = A2_APPROX_USE_THIRD_SHELL != 0;
constexpr int kApproxThirdShellRadius = A2_APPROX_THIRD_SHELL_RADIUS;
constexpr int kApproxOwnCellAcceptNumerator = A2_APPROX_OWN_CELL_ACCEPT_NUM;
constexpr int kApproxOwnCellAcceptDenominator = A2_APPROX_OWN_CELL_ACCEPT_DEN;
constexpr int kApproxThirdShellTriggerNumerator = A2_APPROX_THIRD_SHELL_TRIGGER_NUM;
constexpr int kApproxThirdShellTriggerDenominator = A2_APPROX_THIRD_SHELL_TRIGGER_DEN;
constexpr double kVoxelDistancePercentile = 0.60;
constexpr uint64_t kCellHashMulX = 0x9e3779b97f4a7c15ULL;
constexpr uint64_t kCellHashMulY = 0xc2b2ae3d27d4eb4fULL;
constexpr uint64_t kCellHashMulZ = 0x165667b19e3779f9ULL;
using IntensityType = std::uint8_t;
using HistogramCountType = int;
using FlagType = std::uint8_t;
using DistanceType = long long;
constexpr DistanceType kInfiniteDistance = 0x7fffffffffffffffLL;

template <typename T>
__host__ __device__ inline T load_global(const T* ptr, const int index) {
#ifdef __CUDA_ARCH__
    return __ldg(ptr + index);
#else
    return ptr[index];
#endif
}

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

struct DeviceInput {
    int n = 0;
    int* xs = nullptr;
    int* ys = nullptr;
    int* zs = nullptr;
    IntensityType* intensities = nullptr;
};

struct CommonWorkspace {
    int* output = nullptr;
};

struct ExactWorkspace {
    DeviceInput input;
    CommonWorkspace common;
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

struct ApproxDeviceIndex {
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
    FlagType* fallback_flags = nullptr;
    int* fallback_queries = nullptr;
    int* fallback_count = nullptr;
};

struct SpatialIndexBounds {
    long long min_cell_x = 0;
    long long max_cell_x = 0;
    long long min_cell_y = 0;
    long long max_cell_y = 0;
    long long min_cell_z = 0;
    long long max_cell_z = 0;
};

struct SharedKNNIndex {
    SpatialIndex host;
    ApproxDeviceIndex device;
    SpatialIndexBounds bounds;
};

struct KMeansWorkspace {
    int k = 0;
    int* centroid_xs = nullptr;
    int* centroid_ys = nullptr;
    int* centroid_zs = nullptr;
    int* assignments_prev = nullptr;
    int* assignments_next = nullptr;
    int* changed_count = nullptr;
    int* sum_xs = nullptr;
    int* sum_ys = nullptr;
    int* sum_zs = nullptr;
    int* counts = nullptr;
    int* cluster_sizes = nullptr;
    HistogramCountType* cluster_histograms = nullptr;
};

struct TimingBreakdown {
    double preprocess_ms = 0.0;
    double upload_ms = 0.0;
    double kernel_ms = 0.0;
    double d2h_ms = 0.0;
    double end_to_end_ms = 0.0;
};

struct OutputComparison {
    double mae = 0.0;
    int max_abs_error = 0;
    int mismatch_count = 0;
};

struct BenchmarkReport {
    int n = 0;
    int k = 0;
    int t = 0;
    int seq_threads = 1;
    int gpu_host_threads = 0;
    int warmups = 0;
    int repeats = 0;
    double cold_start_ms = 0.0;
    double shared_input_upload_ms = 0.0;
    double shared_knn_preprocess_ms = 0.0;
    double shared_knn_upload_ms = 0.0;
    double exact_cpu_ms = 0.0;
    double exact_gpu_end_to_end_ms = 0.0;
    double exact_gpu_kernel_ms = 0.0;
    double exact_speedup_end_to_end = 0.0;
    double exact_speedup_compute_only = 0.0;
    double approx_gpu_preprocess_ms = 0.0;
    double approx_gpu_upload_ms = 0.0;
    double approx_gpu_end_to_end_ms = 0.0;
    double approx_gpu_kernel_ms = 0.0;
    double approx_speedup_end_to_end_vs_exact_cpu = 0.0;
    double approx_speedup_compute_only_vs_exact_cpu = 0.0;
    double kmeans_cpu_ms = 0.0;
    double kmeans_gpu_end_to_end_ms = 0.0;
    double kmeans_gpu_kernel_ms = 0.0;
    double kmeans_speedup_end_to_end = 0.0;
    double kmeans_speedup_compute_only = 0.0;
    int approx_fallback_count = 0;
    OutputComparison exact_comparison;
    OutputComparison approx_comparison;
    OutputComparison kmeans_comparison;
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
    const int x,
    const int y,
    const int z,
    const int other_x,
    const int other_y,
    const int other_z) {
    const DistanceType dx = static_cast<DistanceType>(x) - other_x;
    const DistanceType dy = static_cast<DistanceType>(y) - other_y;
    const DistanceType dz = static_cast<DistanceType>(z) - other_z;
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

__host__ __device__ inline void accumulate_intensity_stats(
    const int candidate_intensity,
    int* min_intensity,
    int* count_min,
    int* count_leq_center,
    const int center_intensity) {
    if (candidate_intensity < *min_intensity) {
        *min_intensity = candidate_intensity;
        *count_min = 1;
    } else if (candidate_intensity == *min_intensity) {
        ++(*count_min);
    }
    if (candidate_intensity <= center_intensity) {
        ++(*count_leq_center);
    }
}

__host__ __device__ inline int remap_intensity_from_stats(
    const int center_intensity,
    const int neighborhood_size,
    const int count_min,
    const int count_leq_center) {
    if (neighborhood_size == count_min) {
        return center_intensity;
    }

    const int numerator = count_leq_center - count_min;
    if (numerator <= 0) {
        return 0;
    }

    const int denominator = neighborhood_size - count_min;
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

__host__ __device__ inline int scaled_threshold_count(
    const int base_count,
    const int numerator,
    const int denominator) {
    if (numerator <= 0) {
        return 0;
    }
    if (denominator <= 0) {
        return base_count;
    }
    return (base_count * numerator + denominator - 1) / denominator;
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
    if (data.t <= 0 || data.t > kMaxT) {
        throw std::runtime_error("T must be in [1, 50].");
    }
    if (data.k >= data.n) {
        throw std::runtime_error("KNN requires k < n because the point itself is excluded from the neighbor set.");
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

std::string join_path(const std::string& directory, const std::string& file_name) {
    if (directory.empty() || directory == ".") {
        return file_name;
    }
    if (!directory.empty() && directory.back() == '/') {
        return directory + file_name;
    }
    return directory + "/" + file_name;
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

// Exact KNN CPU/GPU
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

double elapsed_event_ms(cudaEvent_t start, cudaEvent_t stop) {
    float milliseconds = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    return static_cast<double>(milliseconds);
}

// Approximate KNN CPU/GPU

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
                    const int candidate_index = load_global(sorted_original_indices, position);
                    if (candidate_index == query_index) {
                        continue;
                    }
                    ++(*candidate_count);
                    const int candidate_x = load_global(sorted_xs, position);
                    const int candidate_y = load_global(sorted_ys, position);
                    const int candidate_z = load_global(sorted_zs, position);
                    const DistanceType distance = squared_distance(
                        query_x,
                        query_y,
                        query_z,
                        candidate_x,
                        candidate_y,
                        candidate_z);
                    insert_candidate(
                        distance,
                        candidate_x,
                        candidate_y,
                        candidate_z,
                        candidate_index,
                        load_global(sorted_intensities, position),
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
    const int* __restrict__ cell_hash_values,
    const int cell_hash_mask,
    const int* __restrict__ cell_starts,
    const int* __restrict__ cell_ends,
    const int num_cells,
    const int n,
    const int k,
    const double inv_cell_size,
    int* __restrict__ output_intensities,
    FlagType* __restrict__ fallback_flags) {
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

    const int query_x = load_global(query_xs, point_index);
    const int query_y = load_global(query_ys, point_index);
    const int query_z = load_global(query_zs, point_index);
    const IntensityType center_intensity = load_global(query_intensities, point_index);
    const long long query_cell_x = coordinate_to_cell(query_x, inv_cell_size);
    const long long query_cell_y = coordinate_to_cell(query_y, inv_cell_size);
    const long long query_cell_z = coordinate_to_cell(query_z, inv_cell_size);

    int candidate_count = 0;
    const int shell0_accept_count = scaled_threshold_count(
        k,
        kApproxOwnCellAcceptNumerator,
        kApproxOwnCellAcceptDenominator);
    const int shell2_trigger_count = scaled_threshold_count(
        k,
        kApproxThirdShellTriggerNumerator,
        kApproxThirdShellTriggerDenominator);

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

    if (kApproxUseSecondShell && candidate_count < shell0_accept_count) {
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
    }

    if (kApproxUseThirdShell && candidate_count < shell2_trigger_count) {
        scan_shell(
            query_cell_x,
            query_cell_y,
            query_cell_z,
            query_x,
            query_y,
            query_z,
            point_index,
            kApproxThirdShellRadius,
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
    }

    const bool resolved = candidate_count >= k;

    int min_intensity = static_cast<int>(center_intensity);
    int count_min = 1;
    int count_leq_center = 1;
    for (int neighbor = 0; neighbor < k; ++neighbor) {
        accumulate_intensity_stats(
            static_cast<int>(best_intensities[neighbor]),
            &min_intensity,
            &count_min,
            &count_leq_center,
            static_cast<int>(center_intensity));
    }
    output_intensities[point_index] = remap_intensity_from_stats(
        static_cast<int>(center_intensity),
        k + 1,
        count_min,
        count_leq_center);
    fallback_flags[point_index] = resolved ? 0 : 1;
}

template <int BlockSize>
__global__ void collect_fallback_queries_kernel(
    const FlagType* __restrict__ fallback_flags,
    const int n,
    int* __restrict__ fallback_count,
    int* __restrict__ fallback_queries) {
    const int point_index = blockIdx.x * BlockSize + threadIdx.x;
    if (point_index >= n || fallback_flags[point_index] == 0) {
        return;
    }
    const int position = atomicAdd(fallback_count, 1);
    fallback_queries[position] = point_index;
}

template <int BlockSize>
__global__ void exact_fallback_kernel(
    const int* __restrict__ xs,
    const int* __restrict__ ys,
    const int* __restrict__ zs,
    const IntensityType* __restrict__ intensities,
    const int n,
    const int k,
    const int fallback_count,
    const int* __restrict__ fallback_queries,
    int* __restrict__ output_intensities) {
    __shared__ int tile_xs[BlockSize];
    __shared__ int tile_ys[BlockSize];
    __shared__ int tile_zs[BlockSize];
    __shared__ IntensityType tile_intensities[BlockSize];

    const int fallback_id = blockIdx.x * BlockSize + threadIdx.x;
    const bool active = fallback_id < fallback_count;

    DistanceType best_distances[kMaxK];
    int best_indices[kMaxK];
    IntensityType best_intensities[kMaxK];

    int point_index = 0;
    int query_x = 0;
    int query_y = 0;
    int query_z = 0;
    IntensityType center_intensity = 0;

    if (active) {
        point_index = load_global(fallback_queries, fallback_id);
        query_x = load_global(xs, point_index);
        query_y = load_global(ys, point_index);
        query_z = load_global(zs, point_index);
        center_intensity = load_global(intensities, point_index);
        for (int neighbor = 0; neighbor < k; ++neighbor) {
            best_distances[neighbor] = kInfiniteDistance;
            best_indices[neighbor] = -1;
            best_intensities[neighbor] = 0;
        }
    }

    for (int tile_start = 0; tile_start < n; tile_start += BlockSize) {
        const int load_index = tile_start + threadIdx.x;
        if (load_index < n) {
            tile_xs[threadIdx.x] = load_global(xs, load_index);
            tile_ys[threadIdx.x] = load_global(ys, load_index);
            tile_zs[threadIdx.x] = load_global(zs, load_index);
            tile_intensities[threadIdx.x] = load_global(intensities, load_index);
        }
        __syncthreads();

        if (active) {
            const int tile_count = (n - tile_start) < BlockSize ? (n - tile_start) : BlockSize;
            for (int tile_offset = 0; tile_offset < tile_count; ++tile_offset) {
                const int candidate_index = tile_start + tile_offset;
                if (candidate_index == point_index) {
                    continue;
                }
                const DistanceType distance = squared_distance(
                    query_x,
                    query_y,
                    query_z,
                    tile_xs[tile_offset],
                    tile_ys[tile_offset],
                    tile_zs[tile_offset]);
                insert_candidate(
                    distance,
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

    int min_intensity = static_cast<int>(center_intensity);
    int count_min = 1;
    int count_leq_center = 1;
    for (int neighbor = 0; neighbor < k; ++neighbor) {
        accumulate_intensity_stats(
            static_cast<int>(best_intensities[neighbor]),
            &min_intensity,
            &count_min,
            &count_leq_center,
            static_cast<int>(center_intensity));
    }
    output_intensities[point_index] = remap_intensity_from_stats(
        static_cast<int>(center_intensity),
        k + 1,
        count_min,
        count_leq_center);
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

ApproxDeviceIndex upload_spatial_index(const InputData& data, const SpatialIndex& index, double* upload_ms) {
    ApproxDeviceIndex device_index;
    device_index.n = data.n;
    device_index.num_cells = static_cast<int>(index.cell_xs.size());
    device_index.cell_size = index.cell_size;
    device_index.inv_cell_size = index.inv_cell_size;
    device_index.cell_hash_mask = index.cell_hash_mask;

    const std::size_t point_bytes = static_cast<std::size_t>(data.n) * sizeof(int);
    const std::size_t intensity_bytes = static_cast<std::size_t>(data.n) * sizeof(IntensityType);
    const std::size_t index_bytes = static_cast<std::size_t>(data.n) * sizeof(int);
    const std::size_t flag_bytes = static_cast<std::size_t>(data.n) * sizeof(FlagType);
    const std::size_t query_bytes = static_cast<std::size_t>(data.n) * sizeof(int);
    const std::size_t cell_coord_bytes = static_cast<std::size_t>(device_index.num_cells) * sizeof(long long);
    const std::size_t cell_range_bytes = static_cast<std::size_t>(device_index.num_cells) * sizeof(int);
    const std::size_t hash_bytes = static_cast<std::size_t>(index.cell_hash_values.size()) * sizeof(int);

    const double wall_start = omp_get_wtime();
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
        CUDA_CHECK(cudaMalloc(&device_index.fallback_flags, flag_bytes));
        CUDA_CHECK(cudaMalloc(&device_index.fallback_queries, query_bytes));
        CUDA_CHECK(cudaMalloc(&device_index.fallback_count, sizeof(int)));
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
            device_index.cell_hash_mask = index.cell_hash_mask;
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
        cudaFree(device_index.fallback_flags);
        cudaFree(device_index.fallback_queries);
        cudaFree(device_index.fallback_count);
        cudaFree(device_index.cell_hash_values);
        throw;
    }
    if (upload_ms != nullptr) {
        *upload_ms = (omp_get_wtime() - wall_start) * 1000.0;
    }
    return device_index;
}

void free_spatial_index(ApproxDeviceIndex& device_index) {
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
    cudaFree(device_index.fallback_flags);
    cudaFree(device_index.fallback_queries);
    cudaFree(device_index.fallback_count);
    cudaFree(device_index.cell_hash_values);
    device_index = ApproxDeviceIndex{};
}

SpatialIndexBounds compute_spatial_index_bounds(const SpatialIndex& spatial_index) {
    SpatialIndexBounds bounds;
    if (spatial_index.cell_xs.empty()) {
        return bounds;
    }

    bounds.min_cell_x = bounds.max_cell_x = spatial_index.cell_xs[0];
    bounds.min_cell_y = bounds.max_cell_y = spatial_index.cell_ys[0];
    bounds.min_cell_z = bounds.max_cell_z = spatial_index.cell_zs[0];
    for (std::size_t cell = 1; cell < spatial_index.cell_xs.size(); ++cell) {
        if (spatial_index.cell_xs[cell] < bounds.min_cell_x) {
            bounds.min_cell_x = spatial_index.cell_xs[cell];
        }
        if (spatial_index.cell_xs[cell] > bounds.max_cell_x) {
            bounds.max_cell_x = spatial_index.cell_xs[cell];
        }
        if (spatial_index.cell_ys[cell] < bounds.min_cell_y) {
            bounds.min_cell_y = spatial_index.cell_ys[cell];
        }
        if (spatial_index.cell_ys[cell] > bounds.max_cell_y) {
            bounds.max_cell_y = spatial_index.cell_ys[cell];
        }
        if (spatial_index.cell_zs[cell] < bounds.min_cell_z) {
            bounds.min_cell_z = spatial_index.cell_zs[cell];
        }
        if (spatial_index.cell_zs[cell] > bounds.max_cell_z) {
            bounds.max_cell_z = spatial_index.cell_zs[cell];
        }
    }
    return bounds;
}

SharedKNNIndex build_shared_knn_index(
    const InputData& data,
    const HostArrays& arrays,
    double* preprocess_ms,
    double* upload_ms) {
    SharedKNNIndex index;

    const double preprocess_start = omp_get_wtime();
    index.host = build_spatial_index(data, arrays);
    index.bounds = compute_spatial_index_bounds(index.host);
    if (preprocess_ms != nullptr) {
        *preprocess_ms = (omp_get_wtime() - preprocess_start) * 1000.0;
    }

    index.device = upload_spatial_index(data, index.host, upload_ms);
    return index;
}

void free_shared_knn_index(SharedKNNIndex& index) {
    free_spatial_index(index.device);
    index = SharedKNNIndex{};
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
    const int* __restrict__ cell_hash_values,
    const int cell_hash_mask,
    const int* __restrict__ cell_starts,
    const int* __restrict__ cell_ends,
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

    const int query_x = load_global(query_xs, point_index);
    const int query_y = load_global(query_ys, point_index);
    const int query_z = load_global(query_zs, point_index);
    const IntensityType center_intensity = load_global(query_intensities, point_index);
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

    int min_intensity = static_cast<int>(center_intensity);
    int count_min = 1;
    int count_leq_center = 1;
    for (int neighbor = 0; neighbor < k; ++neighbor) {
        accumulate_intensity_stats(
            static_cast<int>(best_intensities[neighbor]),
            &min_intensity,
            &count_min,
            &count_leq_center,
            static_cast<int>(center_intensity));
    }
    output_intensities[point_index] = remap_intensity_from_stats(
        static_cast<int>(center_intensity),
        k + 1,
        count_min,
        count_leq_center);
}

std::vector<int> compute_exact_knn_gpu(
    const InputData& data,
    const HostArrays& arrays,
    const DeviceInput& device_input,
    const CommonWorkspace& workspace,
    TimingBreakdown* timing,
    const SharedKNNIndex* shared_knn_index) {
    const double wall_start = omp_get_wtime();
    SharedKNNIndex owned_knn_index;
    const SharedKNNIndex* knn_index = shared_knn_index;
    if (knn_index == nullptr) {
        owned_knn_index = build_shared_knn_index(
            data,
            arrays,
            timing != nullptr ? &timing->preprocess_ms : nullptr,
            timing != nullptr ? &timing->upload_ms : nullptr);
        knn_index = &owned_knn_index;
    } else if (timing != nullptr) {
        timing->preprocess_ms = 0.0;
        timing->upload_ms = 0.0;
    }

    const ApproxDeviceIndex& device_index = knn_index->device;
    const SpatialIndexBounds& bounds = knn_index->bounds;
    cudaEvent_t kernel_start = nullptr;
    cudaEvent_t kernel_stop = nullptr;

    std::vector<int> output(data.n, 0);
    const std::size_t output_bytes = static_cast<std::size_t>(data.n) * sizeof(int);
    try {
        CUDA_CHECK(cudaEventCreate(&kernel_start));
        CUDA_CHECK(cudaEventCreate(&kernel_stop));

        const int blocks = (data.n + kExactThreadsPerBlock - 1) / kExactThreadsPerBlock;
        CUDA_CHECK(cudaEventRecord(kernel_start));
        exact_grid_knn_equalize_kernel<kExactThreadsPerBlock><<<blocks, kExactThreadsPerBlock>>>(
            device_input.xs,
            device_input.ys,
            device_input.zs,
            device_input.intensities,
            device_index.sorted_xs,
            device_index.sorted_ys,
            device_index.sorted_zs,
            device_index.sorted_intensities,
            device_index.sorted_original_indices,
            device_index.cell_xs,
            device_index.cell_ys,
            device_index.cell_zs,
            device_index.cell_hash_values,
            device_index.cell_hash_mask,
            device_index.cell_starts,
            device_index.cell_ends,
            device_index.num_cells,
            data.n,
            data.k,
            device_index.cell_size,
            device_index.inv_cell_size,
            bounds.min_cell_x,
            bounds.max_cell_x,
            bounds.min_cell_y,
            bounds.max_cell_y,
            bounds.min_cell_z,
            bounds.max_cell_z,
            workspace.output);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(kernel_stop));
        CUDA_CHECK(cudaEventSynchronize(kernel_stop));

        const double d2h_start = omp_get_wtime();
        CUDA_CHECK(cudaMemcpy(output.data(), workspace.output, output_bytes, cudaMemcpyDeviceToHost));
        const double d2h_end = omp_get_wtime();
        if (timing != nullptr) {
            timing->kernel_ms = elapsed_event_ms(kernel_start, kernel_stop);
            timing->d2h_ms = (d2h_end - d2h_start) * 1000.0;
            timing->end_to_end_ms = (d2h_end - wall_start) * 1000.0;
        }
    } catch (...) {
        if (kernel_start != nullptr) {
            cudaEventDestroy(kernel_start);
        }
        if (kernel_stop != nullptr) {
            cudaEventDestroy(kernel_stop);
        }
        if (shared_knn_index == nullptr) {
            free_shared_knn_index(owned_knn_index);
        }
        throw;
    }

    CUDA_CHECK(cudaEventDestroy(kernel_start));
    CUDA_CHECK(cudaEventDestroy(kernel_stop));
    if (shared_knn_index == nullptr) {
        free_shared_knn_index(owned_knn_index);
    }
    return output;
}

std::vector<int> compute_approx_knn_gpu(
    const InputData& data,
    const HostArrays& arrays,
    const DeviceInput& device_input,
    const CommonWorkspace& workspace,
    TimingBreakdown* timing,
    int* fallback_count_out,
    const SharedKNNIndex* shared_knn_index) {
    const double wall_start = omp_get_wtime();
    SharedKNNIndex owned_knn_index;
    const SharedKNNIndex* knn_index = shared_knn_index;
    if (knn_index == nullptr) {
        owned_knn_index = build_shared_knn_index(
            data,
            arrays,
            timing != nullptr ? &timing->preprocess_ms : nullptr,
            timing != nullptr ? &timing->upload_ms : nullptr);
        knn_index = &owned_knn_index;
    } else if (timing != nullptr) {
        timing->preprocess_ms = 0.0;
        timing->upload_ms = 0.0;
    }

    const ApproxDeviceIndex& device_index = knn_index->device;
    const int blocks = (data.n + kApproxThreadsPerBlock - 1) / kApproxThreadsPerBlock;
    cudaEvent_t kernel_start = nullptr;
    cudaEvent_t kernel_stop = nullptr;
    cudaEvent_t fallback_start = nullptr;
    cudaEvent_t fallback_stop = nullptr;

    int fallback_count = 0;
    std::vector<int> output(data.n, 0);
    const std::size_t output_bytes = static_cast<std::size_t>(data.n) * sizeof(int);
    try {
        CUDA_CHECK(cudaEventCreate(&kernel_start));
        CUDA_CHECK(cudaEventCreate(&kernel_stop));
        CUDA_CHECK(cudaMemset(device_index.fallback_count, 0, sizeof(int)));

        CUDA_CHECK(cudaEventRecord(kernel_start));
        approx_knn_equalize_kernel<kApproxThreadsPerBlock><<<blocks, kApproxThreadsPerBlock>>>(
            device_input.xs,
            device_input.ys,
            device_input.zs,
            device_input.intensities,
            device_index.sorted_xs,
            device_index.sorted_ys,
            device_index.sorted_zs,
            device_index.sorted_intensities,
            device_index.sorted_original_indices,
            device_index.cell_xs,
            device_index.cell_ys,
            device_index.cell_zs,
            device_index.cell_hash_values,
            device_index.cell_hash_mask,
            device_index.cell_starts,
            device_index.cell_ends,
            device_index.num_cells,
            data.n,
            data.k,
            device_index.inv_cell_size,
            workspace.output,
            device_index.fallback_flags);
        CUDA_CHECK(cudaGetLastError());

        collect_fallback_queries_kernel<kApproxThreadsPerBlock><<<blocks, kApproxThreadsPerBlock>>>(
            device_index.fallback_flags,
            data.n,
            device_index.fallback_count,
            device_index.fallback_queries);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(kernel_stop));
        CUDA_CHECK(cudaEventSynchronize(kernel_stop));

        CUDA_CHECK(cudaMemcpy(&fallback_count, device_index.fallback_count, sizeof(int), cudaMemcpyDeviceToHost));
        if (fallback_count > 0) {
            CUDA_CHECK(cudaEventCreate(&fallback_start));
            CUDA_CHECK(cudaEventCreate(&fallback_stop));
            const int fallback_blocks = (fallback_count + kApproxThreadsPerBlock - 1) / kApproxThreadsPerBlock;
            CUDA_CHECK(cudaEventRecord(fallback_start));
            exact_fallback_kernel<kApproxThreadsPerBlock><<<fallback_blocks, kApproxThreadsPerBlock>>>(
                device_input.xs,
                device_input.ys,
                device_input.zs,
                device_input.intensities,
                data.n,
                data.k,
                fallback_count,
                device_index.fallback_queries,
                workspace.output);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaEventRecord(fallback_stop));
            CUDA_CHECK(cudaEventSynchronize(fallback_stop));
            if (timing != nullptr) {
                timing->kernel_ms = elapsed_event_ms(kernel_start, kernel_stop) +
                                    elapsed_event_ms(fallback_start, fallback_stop);
            }
        } else if (timing != nullptr) {
            timing->kernel_ms = elapsed_event_ms(kernel_start, kernel_stop);
        }

        const double d2h_start = omp_get_wtime();
        CUDA_CHECK(cudaMemcpy(output.data(), workspace.output, output_bytes, cudaMemcpyDeviceToHost));
        const double d2h_end = omp_get_wtime();
        if (timing != nullptr) {
            timing->d2h_ms = (d2h_end - d2h_start) * 1000.0;
            timing->end_to_end_ms = (d2h_end - wall_start) * 1000.0;
        }
    } catch (...) {
        if (kernel_start != nullptr) {
            cudaEventDestroy(kernel_start);
        }
        if (kernel_stop != nullptr) {
            cudaEventDestroy(kernel_stop);
        }
        if (fallback_start != nullptr) {
            cudaEventDestroy(fallback_start);
        }
        if (fallback_stop != nullptr) {
            cudaEventDestroy(fallback_stop);
        }
        if (shared_knn_index == nullptr) {
            free_shared_knn_index(owned_knn_index);
        }
        throw;
    }

    if (fallback_count_out != nullptr) {
        *fallback_count_out = fallback_count;
    }

    CUDA_CHECK(cudaEventDestroy(kernel_start));
    CUDA_CHECK(cudaEventDestroy(kernel_stop));
    if (fallback_start != nullptr) {
        CUDA_CHECK(cudaEventDestroy(fallback_start));
    }
    if (fallback_stop != nullptr) {
        CUDA_CHECK(cudaEventDestroy(fallback_stop));
    }
    if (shared_knn_index == nullptr) {
        free_shared_knn_index(owned_knn_index);
    }
    return output;
}

// K-means CPU/GPU
int assign_cluster_cpu(
    const int x,
    const int y,
    const int z,
    const std::vector<int>& centroid_xs,
    const std::vector<int>& centroid_ys,
    const std::vector<int>& centroid_zs,
    const int k) {
    DistanceType best_distance = 0;
    int best_cluster = -1;
    int best_x = 0;
    int best_y = 0;
    int best_z = 0;
    for (int cluster = 0; cluster < k; ++cluster) {
        const DistanceType distance = squared_distance(
            x,
            y,
            z,
            centroid_xs[cluster],
            centroid_ys[cluster],
            centroid_zs[cluster]);
        if (best_cluster < 0 ||
            better_candidate(
                distance,
                centroid_xs[cluster],
                centroid_ys[cluster],
                centroid_zs[cluster],
                cluster,
                best_distance,
                best_cluster,
                centroid_xs.data(),
                centroid_ys.data(),
                centroid_zs.data())) {
            best_distance = distance;
            best_cluster = cluster;
            best_x = centroid_xs[cluster];
            best_y = centroid_ys[cluster];
            best_z = centroid_zs[cluster];
            (void)best_x;
            (void)best_y;
            (void)best_z;
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
    std::vector<HistogramCountType> cluster_histograms(static_cast<std::size_t>(data.k) * kIntensityLevels, 0);
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
        const int x = load_global(xs, point_index);
        const int y = load_global(ys, point_index);
        const int z = load_global(zs, point_index);
        DistanceType best_distance = 0;
        int best_cluster = -1;

        for (int cluster = 0; cluster < k; ++cluster) {
            const DistanceType distance = squared_distance(
                x,
                y,
                z,
                shared_centroid_xs[cluster],
                shared_centroid_ys[cluster],
                shared_centroid_zs[cluster]);
            if (best_cluster < 0 ||
                better_candidate(
                    distance,
                    shared_centroid_xs[cluster],
                    shared_centroid_ys[cluster],
                    shared_centroid_zs[cluster],
                    cluster,
                    best_distance,
                    best_cluster,
                    shared_centroid_xs,
                    shared_centroid_ys,
                    shared_centroid_zs)) {
                best_distance = distance;
                best_cluster = cluster;
            }
        }

        next_assignments[point_index] = best_cluster;
        const int previous_assignment = load_global(previous_assignments, point_index);
        if (previous_assignment != best_cluster) {
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
        if (block_counts[cluster] > 0) {
            atomicAdd(&sum_xs[cluster], block_sum_xs[cluster]);
            atomicAdd(&sum_ys[cluster], block_sum_ys[cluster]);
            atomicAdd(&sum_zs[cluster], block_sum_zs[cluster]);
            atomicAdd(&counts[cluster], block_counts[cluster]);
        }
    }
}

// Fused kernel: assignment + accumulation in one pass over points,
// saving one full global-memory read of xs/ys/zs per K-means iteration.
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
        const int x = load_global(xs, point_index);
        const int y = load_global(ys, point_index);
        const int z = load_global(zs, point_index);
        DistanceType best_distance = 0;

        for (int cluster = 0; cluster < k; ++cluster) {
            const DistanceType distance = squared_distance(
                x, y, z,
                shared_centroid_xs[cluster],
                shared_centroid_ys[cluster],
                shared_centroid_zs[cluster]);
            if (best_cluster < 0 ||
                better_candidate(
                    distance,
                    shared_centroid_xs[cluster],
                    shared_centroid_ys[cluster],
                    shared_centroid_zs[cluster],
                    cluster,
                    best_distance,
                    best_cluster,
                    shared_centroid_xs,
                    shared_centroid_ys,
                    shared_centroid_zs)) {
                best_distance = distance;
                best_cluster = cluster;
            }
        }

        next_assignments[point_index] = best_cluster;
        const int previous_assignment = load_global(previous_assignments, point_index);
        if (previous_assignment != best_cluster) {
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
        if (block_counts[cluster] > 0) {
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
    const int cluster = load_global(assignments, point_index);
    const int intensity = static_cast<int>(load_global(intensities, point_index));
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
    const int cluster = load_global(assignments, point_index);
    const HistogramCountType* histogram = cluster_histograms + cluster * kIntensityLevels;
    output_intensities[point_index] = remap_intensity(
        histogram,
        static_cast<int>(load_global(intensities, point_index)),
        cluster_sizes[cluster]);
}

KMeansWorkspace allocate_kmeans_workspace(const InputData& data) {
    KMeansWorkspace workspace;
    workspace.k = data.k;
    const std::size_t centroid_bytes = static_cast<std::size_t>(data.k) * sizeof(int);
    const std::size_t assignment_bytes = static_cast<std::size_t>(data.n) * sizeof(int);
    const std::size_t histogram_bytes = static_cast<std::size_t>(data.k) * kIntensityLevels * sizeof(HistogramCountType);
    try {
        CUDA_CHECK(cudaMalloc(&workspace.centroid_xs, centroid_bytes));
        CUDA_CHECK(cudaMalloc(&workspace.centroid_ys, centroid_bytes));
        CUDA_CHECK(cudaMalloc(&workspace.centroid_zs, centroid_bytes));
        CUDA_CHECK(cudaMalloc(&workspace.assignments_prev, assignment_bytes));
        CUDA_CHECK(cudaMalloc(&workspace.assignments_next, assignment_bytes));
        CUDA_CHECK(cudaMalloc(&workspace.changed_count, sizeof(int)));
        CUDA_CHECK(cudaMalloc(&workspace.sum_xs, centroid_bytes));
        CUDA_CHECK(cudaMalloc(&workspace.sum_ys, centroid_bytes));
        CUDA_CHECK(cudaMalloc(&workspace.sum_zs, centroid_bytes));
        CUDA_CHECK(cudaMalloc(&workspace.counts, centroid_bytes));
        CUDA_CHECK(cudaMalloc(&workspace.cluster_sizes, centroid_bytes));
        CUDA_CHECK(cudaMalloc(&workspace.cluster_histograms, histogram_bytes));
    } catch (...) {
        cudaFree(workspace.centroid_xs);
        cudaFree(workspace.centroid_ys);
        cudaFree(workspace.centroid_zs);
        cudaFree(workspace.assignments_prev);
        cudaFree(workspace.assignments_next);
        cudaFree(workspace.changed_count);
        cudaFree(workspace.sum_xs);
        cudaFree(workspace.sum_ys);
        cudaFree(workspace.sum_zs);
        cudaFree(workspace.counts);
        cudaFree(workspace.cluster_sizes);
        cudaFree(workspace.cluster_histograms);
        throw;
    }
    return workspace;
}

void free_kmeans_workspace(KMeansWorkspace& workspace) {
    cudaFree(workspace.centroid_xs);
    cudaFree(workspace.centroid_ys);
    cudaFree(workspace.centroid_zs);
    cudaFree(workspace.assignments_prev);
    cudaFree(workspace.assignments_next);
    cudaFree(workspace.changed_count);
    cudaFree(workspace.sum_xs);
    cudaFree(workspace.sum_ys);
    cudaFree(workspace.sum_zs);
    cudaFree(workspace.counts);
    cudaFree(workspace.cluster_sizes);
    cudaFree(workspace.cluster_histograms);
    workspace = KMeansWorkspace{};
}

std::vector<int> compute_kmeans_gpu(
    const InputData& data,
    const HostArrays& arrays,
    const DeviceInput& device_input,
    const CommonWorkspace& common,
    KMeansWorkspace& workspace,
    TimingBreakdown* timing) {
    const std::size_t centroid_bytes = static_cast<std::size_t>(data.k) * sizeof(int);
    const std::size_t assignment_bytes = static_cast<std::size_t>(data.n) * sizeof(int);
    const std::size_t histogram_bytes = static_cast<std::size_t>(data.k) * kIntensityLevels * sizeof(HistogramCountType);
    const std::size_t output_bytes = static_cast<std::size_t>(data.n) * sizeof(int);
    const int point_blocks = (data.n + kKMeansThreadsPerBlock - 1) / kKMeansThreadsPerBlock;
    const int centroid_blocks = (data.k + kKMeansThreadsPerBlock - 1) / kKMeansThreadsPerBlock;

    std::vector<int> initial_centroid_xs(data.k);
    std::vector<int> initial_centroid_ys(data.k);
    std::vector<int> initial_centroid_zs(data.k);
    for (int cluster = 0; cluster < data.k; ++cluster) {
        initial_centroid_xs[cluster] = arrays.xs[cluster];
        initial_centroid_ys[cluster] = arrays.ys[cluster];
        initial_centroid_zs[cluster] = arrays.zs[cluster];
    }

    const double wall_start = omp_get_wtime();
    CUDA_CHECK(cudaMemcpy(workspace.centroid_xs, initial_centroid_xs.data(), centroid_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(workspace.centroid_ys, initial_centroid_ys.data(), centroid_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(workspace.centroid_zs, initial_centroid_zs.data(), centroid_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(workspace.assignments_prev, 0xFF, assignment_bytes));

    cudaEvent_t kernel_start;
    cudaEvent_t kernel_stop;
    CUDA_CHECK(cudaEventCreate(&kernel_start));
    CUDA_CHECK(cudaEventCreate(&kernel_stop));
    CUDA_CHECK(cudaEventRecord(kernel_start));

    int* final_assignments = workspace.assignments_prev;
    for (int iteration = 0; iteration < data.t; ++iteration) {
        // Reset cluster-sum accumulators before the fused kernel.
        CUDA_CHECK(cudaMemset(workspace.sum_xs, 0, centroid_bytes));
        CUDA_CHECK(cudaMemset(workspace.sum_ys, 0, centroid_bytes));
        CUDA_CHECK(cudaMemset(workspace.sum_zs, 0, centroid_bytes));
        CUDA_CHECK(cudaMemset(workspace.counts, 0, centroid_bytes));
        // Single fused kernel: assign each point to its nearest centroid AND
        // accumulate the per-cluster coordinate sums in the same pass.
        // This replaces the old assign_clusters_kernel + accumulate_cluster_sums_kernel
        // pair, halving global memory traffic (xs/ys/zs read once, not twice).
        assign_and_accumulate_kernel<kKMeansThreadsPerBlock><<<point_blocks, kKMeansThreadsPerBlock>>>(
            device_input.xs,
            device_input.ys,
            device_input.zs,
            workspace.centroid_xs,
            workspace.centroid_ys,
            workspace.centroid_zs,
            data.n,
            data.k,
            workspace.assignments_prev,
            workspace.assignments_next,
            workspace.changed_count,
            workspace.sum_xs,
            workspace.sum_ys,
            workspace.sum_zs,
            workspace.counts);
        CUDA_CHECK(cudaGetLastError());

        // Recompute centroids from the accumulated sums.
        // No explicit sync needed here: update_centroids_kernel is enqueued in the
        // same default stream as the next iteration's memsets, so CUDA ordering
        // guarantees it finishes before the next fused kernel reads the centroids.
        update_centroids_kernel<<<centroid_blocks, kKMeansThreadsPerBlock>>>(
            data.k,
            workspace.sum_xs,
            workspace.sum_ys,
            workspace.sum_zs,
            workspace.counts,
            workspace.centroid_xs,
            workspace.centroid_ys,
            workspace.centroid_zs);
        CUDA_CHECK(cudaGetLastError());

        std::swap(workspace.assignments_prev, workspace.assignments_next);
        final_assignments = workspace.assignments_prev;
    }

    CUDA_CHECK(cudaMemset(workspace.cluster_sizes, 0, centroid_bytes));
    CUDA_CHECK(cudaMemset(workspace.cluster_histograms, 0, histogram_bytes));
    build_cluster_histograms_kernel<kKMeansThreadsPerBlock><<<point_blocks, kKMeansThreadsPerBlock>>>(
        device_input.intensities,
        final_assignments,
        data.n,
        workspace.cluster_sizes,
        workspace.cluster_histograms);
    CUDA_CHECK(cudaGetLastError());

    remap_clusters_kernel<kKMeansThreadsPerBlock><<<point_blocks, kKMeansThreadsPerBlock>>>(
        device_input.intensities,
        final_assignments,
        workspace.cluster_sizes,
        workspace.cluster_histograms,
        data.n,
        common.output);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(kernel_stop));
    CUDA_CHECK(cudaEventSynchronize(kernel_stop));

    std::vector<int> output(data.n, 0);
    const double d2h_start = omp_get_wtime();
    CUDA_CHECK(cudaMemcpy(output.data(), common.output, output_bytes, cudaMemcpyDeviceToHost));
    const double d2h_end = omp_get_wtime();
    if (timing != nullptr) {
        timing->kernel_ms = elapsed_event_ms(kernel_start, kernel_stop);
        timing->d2h_ms = (d2h_end - d2h_start) * 1000.0;
        timing->end_to_end_ms = (d2h_end - wall_start) * 1000.0;
    }
    CUDA_CHECK(cudaEventDestroy(kernel_start));
    CUDA_CHECK(cudaEventDestroy(kernel_stop));
    return output;
}

// Device setup
DeviceInput upload_device_input(const HostArrays& arrays, double* upload_ms) {
    DeviceInput input;
    input.n = static_cast<int>(arrays.xs.size());
    const std::size_t point_bytes = static_cast<std::size_t>(input.n) * sizeof(int);
    const std::size_t intensity_bytes = static_cast<std::size_t>(input.n) * sizeof(IntensityType);
    const double wall_start = omp_get_wtime();
    try {
        CUDA_CHECK(cudaMalloc(&input.xs, point_bytes));
        CUDA_CHECK(cudaMalloc(&input.ys, point_bytes));
        CUDA_CHECK(cudaMalloc(&input.zs, point_bytes));
        CUDA_CHECK(cudaMalloc(&input.intensities, intensity_bytes));
        CUDA_CHECK(cudaMemcpy(input.xs, arrays.xs.data(), point_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(input.ys, arrays.ys.data(), point_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(input.zs, arrays.zs.data(), point_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(input.intensities, arrays.intensities.data(), intensity_bytes, cudaMemcpyHostToDevice));
    } catch (...) {
        cudaFree(input.xs);
        cudaFree(input.ys);
        cudaFree(input.zs);
        cudaFree(input.intensities);
        throw;
    }
    if (upload_ms != nullptr) {
        *upload_ms = (omp_get_wtime() - wall_start) * 1000.0;
    }
    return input;
}

void free_device_input(DeviceInput& input) {
    cudaFree(input.xs);
    cudaFree(input.ys);
    cudaFree(input.zs);
    cudaFree(input.intensities);
    input = DeviceInput{};
}

CommonWorkspace allocate_common_workspace(const InputData& data) {
    CommonWorkspace workspace;
    const std::size_t output_bytes = static_cast<std::size_t>(data.n) * sizeof(int);
    CUDA_CHECK(cudaMalloc(&workspace.output, output_bytes));
    return workspace;
}

void free_common_workspace(CommonWorkspace& workspace) {
    cudaFree(workspace.output);
    workspace = CommonWorkspace{};
}

OutputComparison compare_outputs(
    const std::vector<int>& reference_output,
    const std::vector<int>& test_output) {
    if (reference_output.size() != test_output.size()) {
        throw std::runtime_error("Output size mismatch during comparison.");
    }
    OutputComparison comparison;
    long long total_abs_error = 0;
    for (std::size_t index = 0; index < reference_output.size(); ++index) {
        const int error = std::abs(reference_output[index] - test_output[index]);
        total_abs_error += error;
        if (error > comparison.max_abs_error) {
            comparison.max_abs_error = error;
        }
        if (error != 0) {
            ++comparison.mismatch_count;
        }
    }
    if (!reference_output.empty()) {
        comparison.mae = static_cast<double>(total_abs_error) / reference_output.size();
    }
    return comparison;
}

double median(std::vector<double> values) {
    if (values.empty()) {
        return 0.0;
    }
    std::sort(values.begin(), values.end());
    return values[values.size() / 2];
}

BenchmarkReport benchmark_case(
    const InputData& data,
    const HostArrays& arrays,
    const int warmups,
    const int repeats) {
    BenchmarkReport report;
    report.n = data.n;
    report.k = data.k;
    report.t = data.t;
    omp_set_dynamic(0);
    const int gpu_host_threads = omp_get_max_threads();
    report.gpu_host_threads = gpu_host_threads;
    report.warmups = warmups;
    report.repeats = repeats;

    const double cold_start_begin = omp_get_wtime();
    CUDA_CHECK(cudaFree(0));
    report.cold_start_ms = (omp_get_wtime() - cold_start_begin) * 1000.0;

    DeviceInput device_input = upload_device_input(arrays, &report.shared_input_upload_ms);
    CommonWorkspace common_workspace = allocate_common_workspace(data);
    KMeansWorkspace kmeans_workspace = allocate_kmeans_workspace(data);

    try {
        std::vector<double> exact_cpu_runs;
        std::vector<double> shared_knn_preprocess_runs;
        std::vector<double> shared_knn_upload_runs;
        std::vector<double> exact_gpu_end_runs;
        std::vector<double> exact_gpu_kernel_runs;
        std::vector<double> approx_gpu_preprocess_runs;
        std::vector<double> approx_gpu_upload_runs;
        std::vector<double> approx_gpu_end_runs;
        std::vector<double> approx_gpu_kernel_runs;
        std::vector<double> kmeans_cpu_runs;
        std::vector<double> kmeans_gpu_end_runs;
        std::vector<double> kmeans_gpu_kernel_runs;
        int last_fallback_count = 0;

        for (int iteration = 0; iteration < warmups + repeats; ++iteration) {
            const bool timed = iteration >= warmups;

            omp_set_num_threads(1);
            const double exact_cpu_start = omp_get_wtime();
            const std::vector<int> exact_cpu_output = compute_exact_knn_cpu(data, arrays);
            const double exact_cpu_ms = (omp_get_wtime() - exact_cpu_start) * 1000.0;
            if (timed) {
                exact_cpu_runs.push_back(exact_cpu_ms);
            }

            omp_set_num_threads(gpu_host_threads);
            double shared_knn_preprocess_ms = 0.0;
            double shared_knn_upload_ms = 0.0;
            SharedKNNIndex shared_knn_index;
            std::vector<int> exact_gpu_output;
            std::vector<int> approx_gpu_output;
            TimingBreakdown exact_gpu_timing;
            TimingBreakdown approx_gpu_timing;
            int fallback_count = 0;
            try {
                shared_knn_index = build_shared_knn_index(
                    data,
                    arrays,
                    &shared_knn_preprocess_ms,
                    &shared_knn_upload_ms);
                exact_gpu_output = compute_exact_knn_gpu(
                    data,
                    arrays,
                    device_input,
                    common_workspace,
                    &exact_gpu_timing,
                    &shared_knn_index);
                approx_gpu_output = compute_approx_knn_gpu(
                    data,
                    arrays,
                    device_input,
                    common_workspace,
                    &approx_gpu_timing,
                    &fallback_count,
                    &shared_knn_index);
                free_shared_knn_index(shared_knn_index);
            } catch (...) {
                free_shared_knn_index(shared_knn_index);
                throw;
            }
            if (timed) {
                shared_knn_preprocess_runs.push_back(shared_knn_preprocess_ms);
                shared_knn_upload_runs.push_back(shared_knn_upload_ms);
                exact_gpu_end_runs.push_back(exact_gpu_timing.end_to_end_ms);
                exact_gpu_kernel_runs.push_back(exact_gpu_timing.kernel_ms);
                approx_gpu_preprocess_runs.push_back(approx_gpu_timing.preprocess_ms);
                approx_gpu_upload_runs.push_back(approx_gpu_timing.upload_ms);
                approx_gpu_end_runs.push_back(approx_gpu_timing.end_to_end_ms);
                approx_gpu_kernel_runs.push_back(approx_gpu_timing.kernel_ms);
                last_fallback_count = fallback_count;
            }

            omp_set_num_threads(1);
            const double kmeans_cpu_start = omp_get_wtime();
            const std::vector<int> kmeans_cpu_output = compute_kmeans_cpu(data, arrays);
            const double kmeans_cpu_ms = (omp_get_wtime() - kmeans_cpu_start) * 1000.0;
            if (timed) {
                kmeans_cpu_runs.push_back(kmeans_cpu_ms);
            }

            omp_set_num_threads(gpu_host_threads);
            TimingBreakdown kmeans_gpu_timing;
            const std::vector<int> kmeans_gpu_output = compute_kmeans_gpu(
                data,
                arrays,
                device_input,
                common_workspace,
                kmeans_workspace,
                &kmeans_gpu_timing);
            if (timed) {
                kmeans_gpu_end_runs.push_back(kmeans_gpu_timing.end_to_end_ms);
                kmeans_gpu_kernel_runs.push_back(kmeans_gpu_timing.kernel_ms);
            }

            if (iteration == warmups + repeats - 1) {
                report.exact_comparison = compare_outputs(exact_cpu_output, exact_gpu_output);
                report.approx_comparison = compare_outputs(exact_cpu_output, approx_gpu_output);
                report.kmeans_comparison = compare_outputs(kmeans_cpu_output, kmeans_gpu_output);
            }
        }

        report.exact_cpu_ms = median(exact_cpu_runs);
        report.shared_knn_preprocess_ms = median(shared_knn_preprocess_runs);
        report.shared_knn_upload_ms = median(shared_knn_upload_runs);
        report.exact_gpu_end_to_end_ms = median(exact_gpu_end_runs);
        report.exact_gpu_kernel_ms = median(exact_gpu_kernel_runs);
        report.exact_speedup_end_to_end = report.exact_gpu_end_to_end_ms > 0.0
            ? report.exact_cpu_ms / report.exact_gpu_end_to_end_ms
            : 0.0;
        report.exact_speedup_compute_only = report.exact_gpu_kernel_ms > 0.0
            ? report.exact_cpu_ms / report.exact_gpu_kernel_ms
            : 0.0;

        report.approx_gpu_preprocess_ms = median(approx_gpu_preprocess_runs);
        report.approx_gpu_upload_ms = median(approx_gpu_upload_runs);
        report.approx_gpu_end_to_end_ms = median(approx_gpu_end_runs);
        report.approx_gpu_kernel_ms = median(approx_gpu_kernel_runs);
        report.approx_speedup_end_to_end_vs_exact_cpu = report.approx_gpu_end_to_end_ms > 0.0
            ? report.exact_cpu_ms / report.approx_gpu_end_to_end_ms
            : 0.0;
        report.approx_speedup_compute_only_vs_exact_cpu = report.approx_gpu_kernel_ms > 0.0
            ? report.exact_cpu_ms / report.approx_gpu_kernel_ms
            : 0.0;
        report.approx_fallback_count = last_fallback_count;

        report.kmeans_cpu_ms = median(kmeans_cpu_runs);
        report.kmeans_gpu_end_to_end_ms = median(kmeans_gpu_end_runs);
        report.kmeans_gpu_kernel_ms = median(kmeans_gpu_kernel_runs);
        report.kmeans_speedup_end_to_end = report.kmeans_gpu_end_to_end_ms > 0.0
            ? report.kmeans_cpu_ms / report.kmeans_gpu_end_to_end_ms
            : 0.0;
        report.kmeans_speedup_compute_only = report.kmeans_gpu_kernel_ms > 0.0
            ? report.kmeans_cpu_ms / report.kmeans_gpu_kernel_ms
            : 0.0;
    } catch (...) {
        free_kmeans_workspace(kmeans_workspace);
        free_common_workspace(common_workspace);
        free_device_input(device_input);
        throw;
    }

    free_kmeans_workspace(kmeans_workspace);
    free_common_workspace(common_workspace);
    free_device_input(device_input);
    return report;
}

BenchmarkReport benchmark_case_gpu_only(
    const InputData& data,
    const HostArrays& arrays,
    const int warmups,
    const int repeats) {
    BenchmarkReport report;
    report.n = data.n;
    report.k = data.k;
    report.t = data.t;
    report.seq_threads = 0;
    omp_set_dynamic(0);
    const int gpu_host_threads = omp_get_max_threads();
    report.gpu_host_threads = gpu_host_threads;
    report.warmups = warmups;
    report.repeats = repeats;

    const double cold_start_begin = omp_get_wtime();
    CUDA_CHECK(cudaFree(0));
    report.cold_start_ms = (omp_get_wtime() - cold_start_begin) * 1000.0;

    DeviceInput device_input = upload_device_input(arrays, &report.shared_input_upload_ms);
    CommonWorkspace common_workspace = allocate_common_workspace(data);
    KMeansWorkspace kmeans_workspace = allocate_kmeans_workspace(data);

    try {
        std::vector<double> shared_knn_preprocess_runs;
        std::vector<double> shared_knn_upload_runs;
        std::vector<double> exact_gpu_end_runs;
        std::vector<double> exact_gpu_kernel_runs;
        std::vector<double> approx_gpu_preprocess_runs;
        std::vector<double> approx_gpu_upload_runs;
        std::vector<double> approx_gpu_end_runs;
        std::vector<double> approx_gpu_kernel_runs;
        std::vector<double> kmeans_gpu_end_runs;
        std::vector<double> kmeans_gpu_kernel_runs;
        int last_fallback_count = 0;

        for (int iteration = 0; iteration < warmups + repeats; ++iteration) {
            const bool timed = iteration >= warmups;
            omp_set_num_threads(gpu_host_threads);

            double shared_knn_preprocess_ms = 0.0;
            double shared_knn_upload_ms = 0.0;
            SharedKNNIndex shared_knn_index;
            TimingBreakdown exact_gpu_timing;
            TimingBreakdown approx_gpu_timing;
            TimingBreakdown kmeans_gpu_timing;
            int fallback_count = 0;
            try {
                shared_knn_index = build_shared_knn_index(
                    data,
                    arrays,
                    &shared_knn_preprocess_ms,
                    &shared_knn_upload_ms);
                (void) compute_exact_knn_gpu(
                    data,
                    arrays,
                    device_input,
                    common_workspace,
                    &exact_gpu_timing,
                    &shared_knn_index);
                (void) compute_approx_knn_gpu(
                    data,
                    arrays,
                    device_input,
                    common_workspace,
                    &approx_gpu_timing,
                    &fallback_count,
                    &shared_knn_index);
                free_shared_knn_index(shared_knn_index);
            } catch (...) {
                free_shared_knn_index(shared_knn_index);
                throw;
            }

            (void) compute_kmeans_gpu(
                data,
                arrays,
                device_input,
                common_workspace,
                kmeans_workspace,
                &kmeans_gpu_timing);

            if (timed) {
                shared_knn_preprocess_runs.push_back(shared_knn_preprocess_ms);
                shared_knn_upload_runs.push_back(shared_knn_upload_ms);
                exact_gpu_end_runs.push_back(exact_gpu_timing.end_to_end_ms);
                exact_gpu_kernel_runs.push_back(exact_gpu_timing.kernel_ms);
                approx_gpu_preprocess_runs.push_back(approx_gpu_timing.preprocess_ms);
                approx_gpu_upload_runs.push_back(approx_gpu_timing.upload_ms);
                approx_gpu_end_runs.push_back(approx_gpu_timing.end_to_end_ms);
                approx_gpu_kernel_runs.push_back(approx_gpu_timing.kernel_ms);
                kmeans_gpu_end_runs.push_back(kmeans_gpu_timing.end_to_end_ms);
                kmeans_gpu_kernel_runs.push_back(kmeans_gpu_timing.kernel_ms);
                last_fallback_count = fallback_count;
            }
        }

        report.shared_knn_preprocess_ms = median(shared_knn_preprocess_runs);
        report.shared_knn_upload_ms = median(shared_knn_upload_runs);
        report.exact_gpu_end_to_end_ms = median(exact_gpu_end_runs);
        report.exact_gpu_kernel_ms = median(exact_gpu_kernel_runs);
        report.approx_gpu_preprocess_ms = median(approx_gpu_preprocess_runs);
        report.approx_gpu_upload_ms = median(approx_gpu_upload_runs);
        report.approx_gpu_end_to_end_ms = median(approx_gpu_end_runs);
        report.approx_gpu_kernel_ms = median(approx_gpu_kernel_runs);
        report.approx_fallback_count = last_fallback_count;
        report.kmeans_gpu_end_to_end_ms = median(kmeans_gpu_end_runs);
        report.kmeans_gpu_kernel_ms = median(kmeans_gpu_kernel_runs);
    } catch (...) {
        free_kmeans_workspace(kmeans_workspace);
        free_common_workspace(common_workspace);
        free_device_input(device_input);
        throw;
    }

    free_kmeans_workspace(kmeans_workspace);
    free_common_workspace(common_workspace);
    free_device_input(device_input);
    return report;
}

void print_benchmark_json(const BenchmarkReport& report) {
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "{";
    std::cout << "\"n\":" << report.n << ',';
    std::cout << "\"k\":" << report.k << ',';
    std::cout << "\"T\":" << report.t << ',';
    std::cout << "\"seq_threads\":" << report.seq_threads << ",";
    std::cout << "\"gpu_host_threads\":" << report.gpu_host_threads << ",";
    std::cout << "\"warmups\":" << report.warmups << ',';
    std::cout << "\"repeats\":" << report.repeats << ',';
    std::cout << "\"cold_start_ms\":" << report.cold_start_ms << ',';
    std::cout << "\"shared_input_upload_ms\":" << report.shared_input_upload_ms << ',';
    std::cout << "\"shared_knn_preprocess_ms\":" << report.shared_knn_preprocess_ms << ',';
    std::cout << "\"shared_knn_upload_ms\":" << report.shared_knn_upload_ms << ',';
    std::cout << "\"exact_cpu_ms\":" << report.exact_cpu_ms << ',';
    std::cout << "\"exact_gpu_end_to_end_ms\":" << report.exact_gpu_end_to_end_ms << ',';
    std::cout << "\"exact_gpu_kernel_ms\":" << report.exact_gpu_kernel_ms << ',';
    std::cout << "\"exact_speedup_end_to_end\":" << report.exact_speedup_end_to_end << ',';
    std::cout << "\"exact_speedup_compute_only\":" << report.exact_speedup_compute_only << ',';
    std::cout << "\"exact_mae\":" << report.exact_comparison.mae << ',';
    std::cout << "\"exact_max_abs_error\":" << report.exact_comparison.max_abs_error << ',';
    std::cout << "\"exact_mismatch_count\":" << report.exact_comparison.mismatch_count << ',';
    std::cout << "\"approx_gpu_preprocess_ms\":" << report.approx_gpu_preprocess_ms << ',';
    std::cout << "\"approx_gpu_upload_ms\":" << report.approx_gpu_upload_ms << ',';
    std::cout << "\"approx_gpu_end_to_end_ms\":" << report.approx_gpu_end_to_end_ms << ',';
    std::cout << "\"approx_gpu_kernel_ms\":" << report.approx_gpu_kernel_ms << ',';
    std::cout << "\"approx_speedup_end_to_end_vs_exact_cpu\":" << report.approx_speedup_end_to_end_vs_exact_cpu << ',';
    std::cout << "\"approx_speedup_compute_only_vs_exact_cpu\":" << report.approx_speedup_compute_only_vs_exact_cpu << ',';
    std::cout << "\"approx_mae_vs_exact\":" << report.approx_comparison.mae << ',';
    std::cout << "\"approx_max_abs_error\":" << report.approx_comparison.max_abs_error << ',';
    std::cout << "\"approx_mismatch_count\":" << report.approx_comparison.mismatch_count << ',';
    std::cout << "\"approx_fallback_count\":" << report.approx_fallback_count << ',';
    std::cout << "\"kmeans_cpu_ms\":" << report.kmeans_cpu_ms << ',';
    std::cout << "\"kmeans_gpu_end_to_end_ms\":" << report.kmeans_gpu_end_to_end_ms << ',';
    std::cout << "\"kmeans_gpu_kernel_ms\":" << report.kmeans_gpu_kernel_ms << ',';
    std::cout << "\"kmeans_speedup_end_to_end\":" << report.kmeans_speedup_end_to_end << ',';
    std::cout << "\"kmeans_speedup_compute_only\":" << report.kmeans_speedup_compute_only << ',';
    std::cout << "\"kmeans_mae\":" << report.kmeans_comparison.mae << ',';
    std::cout << "\"kmeans_max_abs_error\":" << report.kmeans_comparison.max_abs_error << ',';
    std::cout << "\"kmeans_mismatch_count\":" << report.kmeans_comparison.mismatch_count;
    std::cout << "}" << std::endl;
}

bool is_submission_mode(const std::string& mode) {
    return mode == "knn" || mode == "approx_knn" || mode == "kmeans";
}

int run_submission_mode(
    const InputData& data,
    const HostArrays& arrays,
    const std::string& mode,
    const bool validate) {
    double cold_start_ms = 0.0;
    const double cold_start_begin = omp_get_wtime();
    CUDA_CHECK(cudaFree(0));
    cold_start_ms = (omp_get_wtime() - cold_start_begin) * 1000.0;
    (void)cold_start_ms;

    double upload_ms = 0.0;
    DeviceInput device_input = upload_device_input(arrays, &upload_ms);
    CommonWorkspace common_workspace = allocate_common_workspace(data);
    KMeansWorkspace kmeans_workspace;
    SharedKNNIndex shared_knn_index;

    try {
        if (mode == "knn") {
            shared_knn_index = build_shared_knn_index(data, arrays, nullptr, nullptr);
            const std::vector<int> exact_output = compute_exact_knn_gpu(
                data,
                arrays,
                device_input,
                common_workspace,
                nullptr,
                &shared_knn_index);
            write_output("knn.txt", data, exact_output);

            if (validate) {
                const std::vector<int> exact_cpu_output = compute_exact_knn_cpu(data, arrays);
                const OutputComparison exact_cmp = compare_outputs(exact_cpu_output, exact_output);
                std::cout << "Exact KNN mismatches: " << exact_cmp.mismatch_count
                          << ", MAE=" << exact_cmp.mae << '\n';
                if (exact_cmp.mismatch_count != 0) {
                    free_shared_knn_index(shared_knn_index);
                    free_common_workspace(common_workspace);
                    free_device_input(device_input);
                    return 1;
                }
            }
        } else if (mode == "approx_knn") {
            shared_knn_index = build_shared_knn_index(data, arrays, nullptr, nullptr);
            int fallback_count = 0;
            const std::vector<int> approx_output = compute_approx_knn_gpu(
                data,
                arrays,
                device_input,
                common_workspace,
                nullptr,
                &fallback_count,
                &shared_knn_index);
            write_output("approx_knn.txt", data, approx_output);

            if (validate) {
                const std::vector<int> exact_cpu_output = compute_exact_knn_cpu(data, arrays);
                const OutputComparison approx_cmp = compare_outputs(exact_cpu_output, approx_output);
                std::cout << "Approx KNN mismatches: " << approx_cmp.mismatch_count
                          << ", MAE=" << approx_cmp.mae << '\n';
                if (approx_cmp.mae > 3.0) {
                    free_shared_knn_index(shared_knn_index);
                    free_common_workspace(common_workspace);
                    free_device_input(device_input);
                    return 1;
                }
            }
        } else if (mode == "kmeans") {
            kmeans_workspace = allocate_kmeans_workspace(data);
            const std::vector<int> kmeans_output = compute_kmeans_gpu(
                data,
                arrays,
                device_input,
                common_workspace,
                kmeans_workspace,
                nullptr);
            write_output("kmeans.txt", data, kmeans_output);

            if (validate) {
                const std::vector<int> kmeans_cpu_output = compute_kmeans_cpu(data, arrays);
                const OutputComparison kmeans_cmp = compare_outputs(kmeans_cpu_output, kmeans_output);
                std::cout << "KMeans mismatches: " << kmeans_cmp.mismatch_count
                          << ", MAE=" << kmeans_cmp.mae << '\n';
                if (kmeans_cmp.mismatch_count != 0) {
                    free_kmeans_workspace(kmeans_workspace);
                    free_common_workspace(common_workspace);
                    free_device_input(device_input);
                    return 1;
                }
            }
        } else {
            throw std::runtime_error("Unknown mode: " + mode);
        }
    } catch (...) {
        free_shared_knn_index(shared_knn_index);
        free_kmeans_workspace(kmeans_workspace);
        free_common_workspace(common_workspace);
        free_device_input(device_input);
        throw;
    }

    free_shared_knn_index(shared_knn_index);
    free_kmeans_workspace(kmeans_workspace);
    free_common_workspace(common_workspace);
    free_device_input(device_input);
    return 0;
}

int run_gpu_pipeline(
    const InputData& data,
    const HostArrays& arrays,
    const std::string& output_directory,
    const bool validate) {
    double cold_start_ms = 0.0;
    const double cold_start_begin = omp_get_wtime();
    CUDA_CHECK(cudaFree(0));
    cold_start_ms = (omp_get_wtime() - cold_start_begin) * 1000.0;
    (void)cold_start_ms;

    double upload_ms = 0.0;
    DeviceInput device_input = upload_device_input(arrays, &upload_ms);
    CommonWorkspace common_workspace = allocate_common_workspace(data);
    KMeansWorkspace kmeans_workspace = allocate_kmeans_workspace(data);

    SharedKNNIndex shared_knn_index;
    try {
        shared_knn_index = build_shared_knn_index(data, arrays, nullptr, nullptr);
        const std::vector<int> exact_output = compute_exact_knn_gpu(
            data,
            arrays,
            device_input,
            common_workspace,
            nullptr,
            &shared_knn_index);
        const std::vector<int> approx_output = compute_approx_knn_gpu(
            data,
            arrays,
            device_input,
            common_workspace,
            nullptr,
            nullptr,
            &shared_knn_index);
        const std::vector<int> kmeans_output = compute_kmeans_gpu(data, arrays, device_input, common_workspace, kmeans_workspace, nullptr);

        write_output(join_path(output_directory, "knn.txt"), data, exact_output);
        write_output(join_path(output_directory, "approx_knn.txt"), data, approx_output);
        write_output(join_path(output_directory, "kmeans.txt"), data, kmeans_output);

        if (validate) {
            const std::vector<int> exact_cpu_output = compute_exact_knn_cpu(data, arrays);
            const std::vector<int> kmeans_cpu_output = compute_kmeans_cpu(data, arrays);
            const OutputComparison exact_cmp = compare_outputs(exact_cpu_output, exact_output);
            const OutputComparison approx_cmp = compare_outputs(exact_cpu_output, approx_output);
            const OutputComparison kmeans_cmp = compare_outputs(kmeans_cpu_output, kmeans_output);

            std::cout << "Exact KNN mismatches: " << exact_cmp.mismatch_count
                      << ", MAE=" << exact_cmp.mae << '\n';
            std::cout << "Approx KNN mismatches: " << approx_cmp.mismatch_count
                      << ", MAE=" << approx_cmp.mae << '\n';
            std::cout << "KMeans mismatches: " << kmeans_cmp.mismatch_count
                      << ", MAE=" << kmeans_cmp.mae << '\n';

            if (exact_cmp.mismatch_count != 0 || kmeans_cmp.mismatch_count != 0 || approx_cmp.mae > 3.0) {
                free_shared_knn_index(shared_knn_index);
                free_kmeans_workspace(kmeans_workspace);
                free_common_workspace(common_workspace);
                free_device_input(device_input);
                return 1;
            }
        }
        free_shared_knn_index(shared_knn_index);
    } catch (...) {
        free_shared_knn_index(shared_knn_index);
        free_kmeans_workspace(kmeans_workspace);
        free_common_workspace(common_workspace);
        free_device_input(device_input);
        throw;
    }

    free_kmeans_workspace(kmeans_workspace);
    free_common_workspace(common_workspace);
    free_device_input(device_input);
    return 0;
}

void print_usage(const char* program_name) {
    std::cerr << "Usage: " << program_name << " [--validate] [--benchmark] [--benchmark-gpu-only] [--warmups N] [--repeats N] <input_file> [knn|approx_knn|kmeans|output_directory]\n";
}

}  // namespace

int main(int argc, char** argv) {
    try {
        bool validate = false;
        bool benchmark = false;
        bool benchmark_gpu_only = false;
        int warmups = 1;
        int repeats = 3;
        std::vector<std::string> positional;

        for (int arg_index = 1; arg_index < argc; ++arg_index) {
            const std::string argument = argv[arg_index];
            if (argument == "--validate") {
                validate = true;
            } else if (argument == "--benchmark") {
                benchmark = true;
            } else if (argument == "--benchmark-gpu-only") {
                benchmark_gpu_only = true;
            } else if (argument == "--warmups") {
                if (arg_index + 1 >= argc) {
                    throw std::runtime_error("Missing value for --warmups.");
                }
                warmups = std::stoi(argv[++arg_index]);
            } else if (argument == "--repeats") {
                if (arg_index + 1 >= argc) {
                    throw std::runtime_error("Missing value for --repeats.");
                }
                repeats = std::stoi(argv[++arg_index]);
            } else if (argument == "--help" || argument == "-h") {
                print_usage(argv[0]);
                return 0;
            } else {
                positional.push_back(argument);
            }
        }

        if (warmups < 0 || repeats <= 0) {
            throw std::runtime_error("warmups must be non-negative and repeats must be positive.");
        }
        if (positional.empty() || positional.size() > 2) {
            print_usage(argv[0]);
            return 1;
        }

        const std::string input_path = positional[0];
        const std::string second_argument = positional.size() == 2 ? positional[1] : ".";

        if (benchmark && benchmark_gpu_only) {
            throw std::runtime_error("Choose either --benchmark or --benchmark-gpu-only, not both.");
        }

        const InputData data = read_input(input_path);
        const HostArrays arrays = build_host_arrays(data);

        if (benchmark) {
            const BenchmarkReport report = benchmark_case(data, arrays, warmups, repeats);
            print_benchmark_json(report);
            return 0;
        }
        if (benchmark_gpu_only) {
            const BenchmarkReport report = benchmark_case_gpu_only(data, arrays, warmups, repeats);
            print_benchmark_json(report);
            return 0;
        }
        if (positional.size() == 2 && is_submission_mode(second_argument)) {
            return run_submission_mode(data, arrays, second_argument, validate);
        }

        return run_gpu_pipeline(data, arrays, second_argument, validate);
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
