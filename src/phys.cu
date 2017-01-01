#include "ppm.hpp"

#define GLM_FORCE_CUDA
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <thrust/reduce.h>
#include <thrust/execution_policy.h>

__global__ void kCalcInertia(int nFace, const int4 *heFaces, const float *vtxData, glm::mat3 *moiOut, float *massOut, glm::vec3 *cmOut) {
  int fIdx = threadIdx.x + blockIdx.x * blockDim.x;
  if (fIdx >= nFace)
    return;
    
  extern __shared__ glm::mat3 matSM[];
  glm::mat3 &C = matSM[0];
  glm::mat3 &A = matSM[1+threadIdx.x];
  float *pA = glm::value_ptr(A);
  float *pC = glm::value_ptr(C);
  
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      pA[3*i + j] = vtxData[8*heFaces[3*fIdx + i].x + j];
      if (fIdx == 0) pC[3*i + j] = (i == j) ? (1.0f/60) : (1.0f/120);
    }
  }
  float detA = glm::determinant(A);
  
  cmOut[fIdx] = detA*(A[0] + A[1] + A[2])/18.0f;
  massOut[fIdx] = detA/6;
  __syncthreads();
  moiOut[fIdx] = detA * A * C * glm::transpose(A);
}

void PPM::physInit() {
  printf("phys alloc\n");
  glm::mat3 *dev_moi;
  cudaMalloc(&dev_moi, nFace*sizeof(glm::mat3));
  float *dev_mass;
  cudaMalloc(&dev_mass, nFace*sizeof(float));
  glm::vec3 *dev_cm;
  cudaMalloc(&dev_cm, nFace*sizeof(glm::vec3));
  
  printf("phys compute\n");
  dim3 blkDim(1024), blkCnt((nFace + 1023)/1024);
  int nSM = (1+blkDim.x) * sizeof(glm::mat3);
  kCalcInertia<<<blkCnt,blkDim,nSM>>>(nFace, dev_heFaces, dev_vList, dev_moi, dev_mass, dev_cm);
  
  printf("phys reduce\n");
  glm::mat3 moi = thrust::reduce(thrust::device, dev_moi, dev_moi+nFace);
  float mass = thrust::reduce(thrust::device, dev_mass, dev_mass+nFace);
  glm::vec3 cm = thrust::reduce(thrust::device, dev_cm, dev_cm+nFace) / mass;
  moi -= mass*glm::outerProduct(cm,cm);
  moi = glm::mat3(moi[0][0] + moi[1][1] + moi[2][2]) - moi;
  
  cudaFree(dev_moi);
  cudaFree(dev_mass);
  cudaFree(dev_cm);
  
  printf("phys result: %f (%f %f %f)\n", mass, cm.x, cm.y, cm.z);
  printf("%f %f %f\n", moi[0][0], moi[1][0], moi[2][0]);
  printf("%f %f %f\n", moi[0][1], moi[1][1], moi[2][1]);
  printf("%f %f %f\n\n", moi[0][2], moi[1][2], moi[2][2]);
}
