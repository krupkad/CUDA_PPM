#ifndef UTIL_ERROR_H
#define UTIL_ERROR_H

#include <cuda_runtime.h>
#include <cstdio>

// utility to fetch cuda errors and fail
static void checkCUDAError(const char *msg, int line = -1) {
  cudaError_t err = cudaGetLastError();
  if (cudaSuccess != err) {
    if (line >= 0) {
      fprintf(stderr, "Line %d: ", line);
    }
    fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}

#endif
