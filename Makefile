NVCC      = nvcc
NVCCFLAGS = -O3 -std=c++14 -Xcompiler -fopenmp

# Detect GPU arch; default sm_75 (Turing). Override with: make ARCH=sm_86
ARCH ?= sm_75
NVCCFLAGS += -arch=$(ARCH)

all: a2

a2: a2.cu
	$(NVCC) $(NVCCFLAGS) -o a2 a2.cu -lm

clean:
	rm -f a2 knn.txt approx_knn.txt kmeans.txt

.PHONY: all clean
