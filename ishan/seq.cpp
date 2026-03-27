#include <bits/stdc++.h>
#include <vector>
using namespace std;

// FIX: Strict integer coordinates as per assignment spec
struct Point {
  int x, y, z;
  int intensity;
};

int n, k, T;
vector<Point> points;

// FIX: Return long long to prevent integer overflow on large datasets
inline long long distSq(const Point &a, const Point &b) {
  long long dx = a.x - b.x;
  long long dy = a.y - b.y;
  long long dz = a.z - b.z;
  return dx * dx + dy * dy + dz * dz;
}

// Custom comparator for EXACT KNN tie-breaking by lexicographic order of (x, y,
// z)
struct CmpDist {
  bool operator()(const pair<long long, int> &a,
                  const pair<long long, int> &b) const {
    if (a.first != b.first)
      return a.first < b.first;
    const Point &pa = points[a.second];
    const Point &pb = points[b.second];
    if (pa.x != pb.x)
      return pa.x < pb.x;
    if (pa.y != pb.y)
      return pa.y < pb.y;
    return pa.z < pb.z;
  }
};

// Histogram Equalization
int remapIntensity(const vector<int> &neighborhood, int Ii, int m) {
  int hist[256] = {};
  for (int j : neighborhood) {
    hist[points[j].intensity]++;
  }

  int cdf[256] = {};
  cdf[0] = hist[0];
  for (int v = 1; v < 256; v++) {
    cdf[v] = cdf[v - 1] + hist[v];
  }

  int cdf_min = 0;
  for (int v = 0; v < 256; v++) {
    if (cdf[v] > 0) {
      cdf_min = cdf[v];
      break;
    }
  }

  if (m == cdf_min)
    return Ii;

  // FIX: TA formula dictates floor(), not round()
  float num = (float)(cdf[Ii] - cdf_min);
  float den = (float)(m - cdf_min);
  int new_I = floorf((num / den) * 255.0f);

  return max(0, min(255, new_I));
}

// ---- 1. EXACT KNN ----
void runKNN(const string &outfile) {
  ofstream fout(outfile);
  for (int i = 0; i < n; i++) {
    vector<pair<long long, int>> dists;
    dists.reserve(n - 1);

    for (int j = 0; j < n; j++) {
      if (j == i)
        continue; // skip self
      dists.push_back({distSq(points[i], points[j]), j});
    }

    partial_sort(dists.begin(), dists.begin() + k, dists.end(), CmpDist());

    vector<int> neighbors;
    neighbors.reserve(k + 1);
    neighbors.push_back(i); // include self

    for (int m = 0; m < k; m++) {
      neighbors.push_back(dists[m].second);
    }

    int newI = remapIntensity(neighbors, points[i].intensity, k + 1);

    // FIX: Output as strict integers
    fout << points[i].x << " " << points[i].y << " " << points[i].z << " "
         << newI << "\n";
  }
}

// ---- 2. APPROXIMATE KNN (LSH Baseline) ----
void runApproxKNN(const string &outfile) {
  srand(42);
  float rx = (float)rand() / RAND_MAX - 0.5f;
  float ry = (float)rand() / RAND_MAX - 0.5f;
  float rz = (float)rand() / RAND_MAX - 0.5f;
  float norm = sqrt(rx * rx + ry * ry + rz * rz);
  rx /= norm;
  ry /= norm;
  rz /= norm;

  vector<pair<float, int>> proj(n);
  for (int i = 0; i < n; i++) {
    float p = points[i].x * rx + points[i].y * ry + points[i].z * rz;
    proj[i] = {p, i};
  }
  sort(proj.begin(), proj.end());

  vector<int> pos(n);
  for (int s = 0; s < n; s++)
    pos[proj[s].second] = s;

  int W = min(n, 3 * k + 1);
  ofstream fout(outfile);

  for (int i = 0; i < n; i++) {
    int s = pos[i];
    int lo = max(0, s - W / 2), hi = min(n - 1, s + W / 2);

    vector<pair<long long, int>> cands;
    for (int c = lo; c <= hi; c++) {
      int j = proj[c].second;
      if (j == i)
        continue;
      cands.push_back({distSq(points[i], points[j]), j});
    }

    int take = min(k, (int)cands.size());
    partial_sort(cands.begin(), cands.begin() + take, cands.end(), CmpDist());

    vector<int> neighbors;
    neighbors.reserve(take + 1);
    neighbors.push_back(i);
    for (int m = 0; m < take; m++) {
      neighbors.push_back(cands[m].second);
    }

    int newI = remapIntensity(neighbors, points[i].intensity, k + 1);
    fout << points[i].x << " " << points[i].y << " " << points[i].z << " "
         << newI << "\n";
  }
}

// ---- 3. K-MEANS ----
void runKMeans(const string &outfile) {
  // FIX: Centroids must be integers
  vector<Point> centroids(k);
  for (int c = 0; c < k; c++) {
    centroids[c] = points[c];
  }

  vector<int> assign(n, 0);

  for (int iter = 0; iter < T; iter++) {
    bool changed = false;

    // Step A: Assignment
    for (int i = 0; i < n; i++) {
      long long best_dist = LLONG_MAX;
      int bestc = 0;

      for (int c = 0; c < k; c++) {
        long long d = distSq(points[i], centroids[c]);

        if (d < best_dist) {
          best_dist = d;
          bestc = c;
        }
        // FIX: K-Means Lexical Tie-Breaking on Centroid Coordinates
        else if (d == best_dist) {
          if (centroids[c].x < centroids[bestc].x ||
              (centroids[c].x == centroids[bestc].x &&
               centroids[c].y < centroids[bestc].y) ||
              (centroids[c].x == centroids[bestc].x &&
               centroids[c].y == centroids[bestc].y &&
               centroids[c].z < centroids[bestc].z)) {
            bestc = c;
          }
        }
      }

      if (assign[i] != bestc) {
        assign[i] = bestc;
        changed = true;
      }
    }

    if (!changed)
      break;

    // Step B: Update Centroids (Using long long for sums to prevent overflow)
    vector<long long> sx(k, 0), sy(k, 0), sz(k, 0);
    vector<int> cnt(k, 0);

    for (int i = 0; i < n; i++) {
      int c = assign[i];
      sx[c] += points[i].x;
      sy[c] += points[i].y;
      sz[c] += points[i].z;
      cnt[c]++;
    }

    for (int c = 0; c < k; c++) {
      if (cnt[c] > 0) {
        // FIX: Integer division as per TA clarification
        centroids[c].x = (int)(sx[c] / cnt[c]);
        centroids[c].y = (int)(sy[c] / cnt[c]);
        centroids[c].z = (int)(sz[c] / cnt[c]);
      }
    }
  }

  vector<vector<int>> clusters(k);
  for (int i = 0; i < n; i++) {
    clusters[assign[i]].push_back(i);
  }

  ofstream fout(outfile);
  for (int i = 0; i < n; i++) {
    const vector<int> &neighborhood = clusters[assign[i]];
    int m = (int)neighborhood.size();
    int newI = remapIntensity(neighborhood, points[i].intensity, m);

    fout << points[i].x << " " << points[i].y << " " << points[i].z << " "
         << newI << "\n";
  }
}

int main(int argc, char *argv[]) {
  if (argc < 2) {
    cerr << "Usage: ./seq input.txt\n";
    return 1;
  }

  ifstream fin(argv[1]);
  fin >> n >> k >> T;
  points.resize(n);
  for (int i = 0; i < n; i++) {
    fin >> points[i].x >> points[i].y >> points[i].z >> points[i].intensity;
  }

  runKNN("knn_seq.txt");
  cerr << "KNN done\n";

  runApproxKNN("approx_knn_seq.txt");
  cerr << "Approx KNN done\n";

  runKMeans("kmeans_seq.txt");
  cerr << "KMeans done\n";

  return 0;
}