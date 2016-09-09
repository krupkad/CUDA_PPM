#include "dcel.hpp"

#define FORCE_GLM_CUDA
#include "glm/glm.hpp"

#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/device_vector.h>

// utility to fetch cuda errors and fail
void checkCUDAError(const char *msg, int line = -1) {
  cudaError_t err = cudaGetLastError();
  if (cudaSuccess != err) {
    if (line >= 0) {
      fprintf(stderr, "Line %d: ", line);
    }
    fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}

// shuffle a list to a given order
template <typename T>
__global__ void kernShuffleToOrder(int N, T *src, T *dst, int *order) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N)
    return;

  int j = order[i];
  dst[i] = src[j];
}

// given a sorted list, find indices of where each block starts/ends
__global__ void kernFindBoundaries(int N, int *arr, int *starts, int *ends) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N)
    return;

  if (arr[i] == 0 || arr[i-1] != arr[i])
    starts[arr[i]] = i;
  if (arr[i] == N-1 || arr[i+1] != arr[i])
    ends[arr[i]] = i+1;
}

// find the corresponding pair for each halfedge. assumes planarity.
__global__ void kernFindHalfEdgePairs(int N, int *heSrcList, int *heDstList, int *hePairList) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= N || j >= N)
    return;

  if (heSrcList[i] == heDstList[j] && heSrcList[j] == heDstList[i])
    hePairList[i] = hePairList[j];
}

// allocate and initialize DCEL data
void DCEL::devInit(int blkDim1d, int blkDim2d) {
  // get the vertex/halfedge counts
  dcel.vCount = vList.size();
  dcel.heCount = heSrcList.size();
  int N = vList.size();
  int M = heSrcList.size();

  // calculate kernel parameters
  dim3 blkCnt1d((M + blkDim1d - 1) / blkDim1d);
  dim3 blkCnt2d((M + blkDim2d - 1) / blkDim2d, (M + blkDim2d - 1) / blkDim2d);
  dim3 blkSize1d(blkDim1d);
  dim3 blkSize2d(blkDim2d, blkDim2d);

  // upload the vertices
  cudaMalloc((void**)&dcel.vList, N*sizeof(glm::vec3));
  cudaMemcpy(dcel.vList, &vList[0], N*sizeof(glm::vec3), cudaMemcpyHostToDevice);

  // upload the edges
  cudaMalloc((void**)&dcel.heSrcList, M*sizeof(int));
  cudaMalloc((void**)&dcel.heDstList, M*sizeof(int));
  cudaMemcpy(dcel.heSrcList, &heSrcList[0], M*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(dcel.heDstList, &heDstList[0], M*sizeof(int), cudaMemcpyHostToDevice);

  // find per-vertex boundaries of the half-edge list.
  // isolated vertices labeled -1.
  cudaMalloc((void**)&dcel.vEdgesStart, M*sizeof(int));
  cudaMalloc((void**)&dcel.vEdgesEnd, M*sizeof(int));
  cudaMemset(dcel.vEdgesStart, -1, M*sizeof(int));
  cudaMemset(dcel.vEdgesEnd, -1, M*sizeof(int));
  kernFindBoundaries<<<blkDim1d, blkSize1d>>>
    (M, dcel.heSrcList, dcel.vEdgesStart, dcel.vEdgesEnd);
  checkCUDAError("boundaries", __LINE__);

  // sort half-edges by origin
  thrust::device_ptr<int> thrust_heSrcList(dcel.heSrcList);
  thrust::device_ptr<int> thrust_heDstList(dcel.heDstList);
  thrust::sort_by_key(thrust_heDstList, thrust_heDstList + M, thrust_heSrcList);

  // find half-edge pairs. unpaired edges labeled -1.
  cudaMalloc((void**)&dcel.hePairList, M*sizeof(int));
  cudaMemset(dcel.hePairList, -1, M*sizeof(int));
  kernFindHalfEdgePairs<<<blkDim2d, blkSize2d>>>
    (M, dcel.heSrcList, dcel.heDstList, dcel.hePairList);
  checkCUDAError("pairs", __LINE__);

  // upload device DCEL struct
  cudaMalloc((void**)&dev_dcel, sizeof(CuDCEL));
  cudaMemcpy(dev_dcel, &dcel, sizeof(CuDCEL), cudaMemcpyHostToDevice);
  checkCUDAError("dcel upload", __LINE__);

  cudaThreadSynchronize();
  checkCUDAError("sync", __LINE__);
}

// free dcel data
void DCEL::devFree() {
  cudaFree(dcel.vList);
  cudaFree(dcel.heSrcList);
  cudaFree(dcel.heDstList);
  cudaFree(dcel.vEdgesStart);
  cudaFree(dcel.vEdgesEnd);
  cudaFree(dev_dcel);
}

void DCEL::visUpdate() {
  /*
  cudaMemcpy(&vList[0], dcel.vList, dcel.vCount*sizeof(glm::vec3), cudaMemcpyDeviceToHost);
  for(int i = 0; i < vList.size(); i++)
    memcpy(&vboData[6*i], &vList[i], 3*sizeof(float));
  checkCUDAError("download", __LINE__);

  glBindBuffer(GL_ARRAY_BUFFER, vboVtx);
  glBufferData(GL_ARRAY_BUFFER, 3*vList.size()*sizeof(float), vboVtxData, GL_STATIC_DRAW);

  cudaThreadSynchronize();
  checkCUDAError("sync", __LINE__);
  */
}

