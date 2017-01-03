#ifndef UTIL_ERROR_H
#define UTIL_ERROR_H

#undef min
#undef max

#include <string>
#include <sstream>
#include <stdexcept>

#include <cuda_runtime.h>
#include <cstdio>

// utility to fetch cuda errors and fail
static void checkCUDAError(const char *msg, int line = -1) {
  cudaError_t err = cudaGetLastError();
  if (cudaSuccess != err) {
	std::stringstream sstr;
	sstr << "CUDA error: ";
    if (line >= 0)
	  sstr << "Line " << line << ":";
    sstr << msg << ": " << cudaGetErrorString(err);
    throw std::runtime_error(sstr.str());
  }
}

#endif
