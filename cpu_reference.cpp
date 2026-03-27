#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <algorithm>

using namespace std;

struct Point {
    double x, y, z;
    int I;
    int id; // To keep track of original order
};

// Calculate Euclidean distance
double get_dist(const Point& a, const Point& b) {
    return sqrt((a.x - b.x)*(a.x - b.x) + (a.y - b.y)*(a.y - b.y) + (a.z - b.z)*(a.z - b.z));
}

// Helper struct for sorting neighbors with tie-breaking: dist -> x -> y -> z
struct Neighbor {
    double dist;
    Point p;
    
    bool operator<(const Neighbor& other) const {
        // Use a small epsilon for floating point comparison
        if (abs(dist - other.dist) > 1e-9) return dist < other.dist;
        if (abs(p.x - other.p.x) > 1e-9) return p.x < other.p.x;
        if (abs(p.y - other.p.y) > 1e-9) return p.y < other.p.y;
        return p.z < other.p.z;
    }
};

// Calculates the new intensity based on the local histogram
int compute_new_intensity(int orig_I, const vector<int>& hist, int m) {
    int cdf = 0;
    int cdf_orig = 0;
    int c_min = -1;
    
    for (int v = 0; v < 256; ++v) {
        if (hist[v] > 0) {
            cdf += hist[v];
            if (c_min == -1) c_min = cdf;
        }
        if (v == orig_I) {
            cdf_orig = cdf;
        }
    }
    
    // Edge case defined in the assignment
    if (m == c_min || c_min == -1) return orig_I;
    
    double num = cdf_orig - c_min;
    double den = m - c_min;
    return floor((num / den) * 255.0);
}

void run_exact_knn(const vector<Point>& points, int k, const string& out_file) {
    int n = points.size();
    vector<int> new_intensities(n);
    
    for (int i = 0; i < n; ++i) {
        vector<Neighbor> neighbors(n);
        for (int j = 0; j < n; ++j) {
            neighbors[j].dist = get_dist(points[i], points[j]);
            neighbors[j].p = points[j];
        }
        
        // Sort to find k-nearest
        sort(neighbors.begin(), neighbors.end());
        
        // Build local histogram for the top k neighbors
        vector<int> hist(256, 0);
        for (int j = 0; j < k; ++j) {
            hist[neighbors[j].p.I]++;
        }
        
        new_intensities[i] = compute_new_intensity(points[i].I, hist, k);
    }
    
    // Write output
    ofstream out(out_file);
    for (int i = 0; i < n; ++i) {
        out << points[i].x << " " << points[i].y << " " << points[i].z << " " << new_intensities[i] << "\n";
    }
    out.close();
}

void run_exact_kmeans(const vector<Point>& points, int k, int T, const string& out_file) {
    int n = points.size();
    
    // Centroids initialization using first k points
    vector<Point> centroids(k);
    for (int i = 0; i < k; ++i) {
        centroids[i] = points[i];
    }
    
    vector<int> assignments(n, -1);
    
    // K-means iterations
    for (int iter = 0; iter < T; ++iter) {
        bool changed = false;
        
        // Step 1: Assign points to nearest centroid
        for (int i = 0; i < n; ++i) {
            int best_cluster = 0;
            double min_dist = get_dist(points[i], centroids[0]);
            
            for (int c = 1; c < k; ++c) {
                double dist = get_dist(points[i], centroids[c]);
                if (dist < min_dist) {
                    min_dist = dist;
                    best_cluster = c;
                }
            }
            
            if (assignments[i] != best_cluster) {
                assignments[i] = best_cluster;
                changed = true;
            }
        }
        
        // Stop early if converged
        if (!changed) break;
        
        // Step 2: Update centroids
        vector<double> sum_x(k, 0.0), sum_y(k, 0.0), sum_z(k, 0.0);
        vector<int> counts(k, 0);
        
        for (int i = 0; i < n; ++i) {
            int c = assignments[i];
            sum_x[c] += points[i].x;
            sum_y[c] += points[i].y;
            sum_z[c] += points[i].z;
            counts[c]++;
        }
        
        for (int c = 0; c < k; ++c) {
            if (counts[c] > 0) {
                centroids[c].x = sum_x[c] / counts[c];
                centroids[c].y = sum_y[c] / counts[c];
                centroids[c].z = sum_z[c] / counts[c];
            }
        }
    }
    
    // Compute histograms per cluster
    vector<vector<int>> cluster_hists(k, vector<int>(256, 0));
    vector<int> cluster_sizes(k, 0);
    
    for (int i = 0; i < n; ++i) {
        int c = assignments[i];
        cluster_hists[c][points[i].I]++;
        cluster_sizes[c]++;
    }
    
    vector<int> new_intensities(n);
    for (int i = 0; i < n; ++i) {
        int c = assignments[i];
        new_intensities[i] = compute_new_intensity(points[i].I, cluster_hists[c], cluster_sizes[c]);
    }
    
    // Write output
    ofstream out(out_file);
    for (int i = 0; i < n; ++i) {
        out << points[i].x << " " << points[i].y << " " << points[i].z << " " << new_intensities[i] << "\n";
    }
    out.close();
}

int main() {
    ifstream in("input.txt");
    if (!in) {
        cerr << "Error: Could not open input.txt\n";
        return 1;
    }
    
    int n, k, T;
    in >> n >> k >> T;
    
    vector<Point> points(n);
    for (int i = 0; i < n; ++i) {
        in >> points[i].x >> points[i].y >> points[i].z >> points[i].I;
        points[i].id = i;
    }
    in.close();
    
    cout << "Running exact KNN..." << endl;
    run_exact_knn(points, k, "seq_knn.txt");
    
    cout << "Running exact K-Means..." << endl;
    run_exact_kmeans(points, k, T, "seq_kmeans.txt");
    
    cout << "Finished! Outputs written to seq_knn.txt and seq_kmeans.txt" << endl;
    
    return 0;
}