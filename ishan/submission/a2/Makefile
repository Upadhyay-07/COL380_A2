NVCC := nvcc
COMMON_NVCCFLAGS := -O3 -Xcompiler -fopenmp
EXACT_NVCCFLAGS := $(COMMON_NVCCFLAGS) -std=c++14
APPROX_NVCCFLAGS := $(COMMON_NVCCFLAGS) -std=c++14
KMEANS_NVCCFLAGS := $(COMMON_NVCCFLAGS) -std=c++17
ASSIGNMENT_NVCCFLAGS := $(COMMON_NVCCFLAGS) -std=c++14

EXACT_TARGET := exact_knn
APPROX_TARGET := approx_knn
KMEANS_TARGET := kmeans
SUBMISSION_TARGET := a2
ASSIGNMENT_TARGET := assignment2
EXACT_SRC := src/exact_knn.cu
APPROX_SRC := src/approx_knn.cu
KMEANS_SRC := src/kmeans.cu
ASSIGNMENT_SRC := src/assignment2.cu

.PHONY: all test test_approx test_kmeans test_assignment2 benchmark_large package_submission clean

all: $(SUBMISSION_TARGET)

$(EXACT_TARGET): $(EXACT_SRC)
	$(NVCC) $(EXACT_NVCCFLAGS) $< -o $@

$(APPROX_TARGET): $(APPROX_SRC)
	$(NVCC) $(APPROX_NVCCFLAGS) $< -o $@

$(KMEANS_TARGET): $(KMEANS_SRC)
	$(NVCC) $(KMEANS_NVCCFLAGS) $< -o $@

$(SUBMISSION_TARGET): $(ASSIGNMENT_SRC)
	$(NVCC) $(ASSIGNMENT_NVCCFLAGS) $< -o $@

$(ASSIGNMENT_TARGET): $(SUBMISSION_TARGET)
	cp $(SUBMISSION_TARGET) $(ASSIGNMENT_TARGET)

test: $(EXACT_TARGET)
	./$(EXACT_TARGET) --validate testdata/small_case.txt
	./$(EXACT_TARGET) --validate testdata/tie_case.txt
	./$(EXACT_TARGET) --validate testdata/self_exclusion_case.txt
	./$(EXACT_TARGET) testdata/lexicographic_tie_case.txt /tmp/exact_lexicographic_tie.txt
	diff -u testdata/lexicographic_tie_expected.txt /tmp/exact_lexicographic_tie.txt
	./$(EXACT_TARGET) testdata/remap_truncation_case.txt /tmp/exact_remap_truncation.txt
	diff -u testdata/remap_truncation_expected.txt /tmp/exact_remap_truncation.txt

test_approx: $(EXACT_TARGET) $(APPROX_TARGET)
	./$(EXACT_TARGET) testdata/small_case.txt /tmp/exact_small.txt
	./$(APPROX_TARGET) testdata/small_case.txt /tmp/approx_small.txt
	python3 mae_loss.py /tmp/exact_small.txt /tmp/approx_small.txt
	./$(EXACT_TARGET) testdata/tie_case.txt /tmp/exact_tie.txt
	./$(APPROX_TARGET) testdata/tie_case.txt /tmp/approx_tie.txt
	python3 mae_loss.py /tmp/exact_tie.txt /tmp/approx_tie.txt
	./$(EXACT_TARGET) testdata/self_exclusion_case.txt /tmp/exact_self.txt
	./$(APPROX_TARGET) testdata/self_exclusion_case.txt /tmp/approx_self.txt
	python3 mae_loss.py /tmp/exact_self.txt /tmp/approx_self.txt
	./$(APPROX_TARGET) testdata/lexicographic_tie_case.txt /tmp/approx_lexicographic_tie.txt
	diff -u testdata/lexicographic_tie_expected.txt /tmp/approx_lexicographic_tie.txt
	./$(APPROX_TARGET) testdata/remap_truncation_case.txt /tmp/approx_remap_truncation.txt
	diff -u testdata/remap_truncation_expected.txt /tmp/approx_remap_truncation.txt

test_kmeans: $(KMEANS_TARGET)
	./$(KMEANS_TARGET) --validate testdata/small_case.txt
	./$(KMEANS_TARGET) --validate testdata/tie_case.txt
	./$(KMEANS_TARGET) testdata/kmeans_tie_case.txt /tmp/kmeans_tie.txt
	diff -u testdata/kmeans_tie_expected.txt /tmp/kmeans_tie.txt
	./$(KMEANS_TARGET) testdata/kmeans_singleton_case.txt /tmp/kmeans_singleton.txt
	diff -u testdata/kmeans_singleton_expected.txt /tmp/kmeans_singleton.txt

test_assignment2: $(ASSIGNMENT_TARGET)
	rm -rf /tmp/assignment2_small /tmp/assignment2_tie /tmp/assignment2_knn_lex /tmp/assignment2_knn_remap /tmp/assignment2_kmeans_tie /tmp/assignment2_kmeans_singleton
	mkdir -p /tmp/assignment2_small /tmp/assignment2_tie /tmp/assignment2_knn_lex /tmp/assignment2_knn_remap /tmp/assignment2_kmeans_tie /tmp/assignment2_kmeans_singleton
	./$(ASSIGNMENT_TARGET) --validate testdata/small_case.txt /tmp/assignment2_small
	./$(ASSIGNMENT_TARGET) --validate testdata/tie_case.txt /tmp/assignment2_tie
	./$(ASSIGNMENT_TARGET) testdata/lexicographic_tie_case.txt /tmp/assignment2_knn_lex
	diff -u testdata/lexicographic_tie_expected.txt /tmp/assignment2_knn_lex/knn.txt
	diff -u testdata/lexicographic_tie_expected.txt /tmp/assignment2_knn_lex/approx_knn.txt
	./$(ASSIGNMENT_TARGET) testdata/remap_truncation_case.txt /tmp/assignment2_knn_remap
	diff -u testdata/remap_truncation_expected.txt /tmp/assignment2_knn_remap/knn.txt
	diff -u testdata/remap_truncation_expected.txt /tmp/assignment2_knn_remap/approx_knn.txt
	./$(ASSIGNMENT_TARGET) testdata/kmeans_tie_case.txt /tmp/assignment2_kmeans_tie
	diff -u testdata/kmeans_tie_expected.txt /tmp/assignment2_kmeans_tie/kmeans.txt
	./$(KMEANS_TARGET) testdata/kmeans_singleton_case.txt /tmp/assignment2_kmeans_singleton/kmeans.txt
	diff -u testdata/kmeans_singleton_expected.txt /tmp/assignment2_kmeans_singleton/kmeans.txt

benchmark_large: $(ASSIGNMENT_TARGET)
	python3 benchmark_large.py


package_submission: report.pdf
	rm -rf submission/a2 submission/a2.zip
	mkdir -p submission/a2
	cp Makefile report.pdf submission/a2/
	cp -r src submission/a2/
	cd submission && zip -qr a2.zip a2

clean:
	rm -f $(EXACT_TARGET) $(APPROX_TARGET) $(KMEANS_TARGET) $(SUBMISSION_TARGET) $(ASSIGNMENT_TARGET) knn.txt approx_knn.txt kmeans.txt benchmark_results.csv benchmark_large_results.csv
