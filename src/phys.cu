#include "ppm.hpp"

#include <glm/gtc/type_ptr.hpp>

//#include <thrust/execution_policy.h>
//#include <thrust/reduce.h>
//#include <thrust/device_ptr.h>

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
      pA[3*i + j] = vtxData[PPM_NVARS*heFaces[3*fIdx + i].x + j];
      if (threadIdx.x == 0) pC[3*i + j] = (i == j) ? (1.0f/60) : (1.0f/120);
    }
  }
  
  float detA = glm::determinant(A);
  float *cmPtr = glm::value_ptr(*cmOut);
  atomicAdd(&cmPtr[fIdx+0], detA*(pA[0]+pA[3]+pA[6])/18.0f);
  atomicAdd(&cmPtr[fIdx+1], detA*(pA[1]+pA[4]+pA[7])/18.0f);
  atomicAdd(&cmPtr[fIdx+2], detA*(pA[2]+pA[5]+pA[8])/18.0f);
  atomicAdd(massOut, detA/6);
  
  __syncthreads();
  float *moiPtr = glm::value_ptr(*moiOut);
  A = detA * A * C * glm::transpose(A);
  for (int i = 0; i < 9; i++)
	atomicAdd(&moiPtr[i], pA[i]);
}

__global__ void kCalcTessInertia(int nFace, const int *tessIdx, const float *tessVtx, glm::mat3 *moiOut, float *massOut, glm::vec3 *cmOut) {
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
      pA[3*i + j] = tessVtx[PPM_NVARS*tessIdx[3*fIdx + i] + j];
      if (threadIdx.x == 0) pC[3*i + j] = (i == j) ? (1.0f/60) : (1.0f/120);
    }
  }
  
  float detA = glm::determinant(A);
  float *cmPtr = glm::value_ptr(*cmOut);
  atomicAdd(&cmPtr[fIdx+0], detA*(pA[0]+pA[3]+pA[6])/18.0f);
  atomicAdd(&cmPtr[fIdx+1], detA*(pA[1]+pA[4]+pA[7])/18.0f);
  atomicAdd(&cmPtr[fIdx+2], detA*(pA[2]+pA[5]+pA[8])/18.0f);
  atomicAdd(massOut, detA/6);
  
  __syncthreads();
  float *moiPtr = glm::value_ptr(*moiOut);
  A = detA * A * C * glm::transpose(A);
  for (int i = 0; i < 9; i++)
	atomicAdd(&moiPtr[i], pA[i]);
}

void PPM::physInit() {
  fprintf(stderr, "phys alloc\n");
  glm::mat3 *dev_moi;
  cudaMalloc(&dev_moi, sizeof(glm::mat3));
  cudaMemset(dev_moi, 0, sizeof(glm::mat3));
  float *dev_mass;
  cudaMalloc(&dev_mass, sizeof(float));
  cudaMemset(dev_mass, 0, sizeof(float));
  glm::vec3 *dev_cm;
  cudaMalloc(&dev_cm, sizeof(glm::vec3));
  cudaMemset(dev_cm, 0, sizeof(glm::vec3));
  
  fprintf(stderr, "phys compute\n");
  dim3 blkDim(256), blkCnt((nFace + 255)/256);
  int nSM = (1+blkDim.x) * sizeof(glm::mat3);
  kCalcInertia<<<blkCnt,blkDim,nSM>>>(nFace, dev_heFaces, dev_vList, dev_moi, dev_mass, dev_cm);
  
  cudaMemcpy(&moi, dev_moi, sizeof(glm::mat3), cudaMemcpyDeviceToHost);
  cudaMemcpy(&cm, dev_cm, sizeof(glm::vec3), cudaMemcpyDeviceToHost);
  cudaMemcpy(&mass, dev_mass, sizeof(float), cudaMemcpyDeviceToHost);

  fprintf(stderr, "phys reduce\n");
  moi -= mass*glm::outerProduct(cm,cm);
  moi = glm::mat3(moi[0][0] + moi[1][1] + moi[2][2]) - moi;
  fprintf(stderr, "phys result: %f (%f %f %f)\n", mass, cm.x, cm.y, cm.z);
  fprintf(stderr, "%f %f %f\n", moi[0][0], moi[1][0], moi[2][0]);
  fprintf(stderr, "%f %f %f\n", moi[0][1], moi[1][1], moi[2][1]);
  fprintf(stderr, "%f %f %f\n\n", moi[0][2], moi[1][2], moi[2][2]);
  
  cudaFree(dev_moi);
  cudaFree(dev_mass);
  cudaFree(dev_cm);
}

void PPM::physTess() {
  fprintf(stderr, "phys alloc\n");
  glm::mat3 *dev_moi;
  cudaMalloc(&dev_moi, sizeof(glm::mat3));
  cudaMemset(dev_moi, 0, sizeof(glm::mat3));
  float *dev_mass;
  cudaMalloc(&dev_mass, sizeof(float));
  cudaMemset(dev_mass, 0, sizeof(float));
  glm::vec3 *dev_cm;
  cudaMalloc(&dev_cm, sizeof(glm::vec3));
  cudaMemset(dev_cm, 0, sizeof(glm::vec3));
  
  fprintf(stderr, "phys compute\n");
  dim3 blkDim(256), blkCnt((nFace + 255)/256);
  int nSM = (1+blkDim.x) * sizeof(glm::mat3);
  kCalcTessInertia<<<blkCnt,blkDim,nSM>>>(nFace, dev_tessIdx, dev_tessVtx, dev_moi, dev_mass, dev_cm);
  
  cudaMemcpy(&moi, dev_moi, sizeof(glm::mat3), cudaMemcpyDeviceToHost);
  cudaMemcpy(&cm, dev_cm, sizeof(glm::vec3), cudaMemcpyDeviceToHost);
  cudaMemcpy(&mass, dev_mass, sizeof(float), cudaMemcpyDeviceToHost);

  fprintf(stderr, "phys reduce\n");
  moi -= mass*glm::outerProduct(cm,cm);
  moi = glm::mat3(moi[0][0] + moi[1][1] + moi[2][2]) - moi;
  fprintf(stderr, "phys result: %f (%f %f %f)\n", mass, cm.x, cm.y, cm.z);
  fprintf(stderr, "%f %f %f\n", moi[0][0], moi[1][0], moi[2][0]);
  fprintf(stderr, "%f %f %f\n", moi[0][1], moi[1][1], moi[2][1]);
  fprintf(stderr, "%f %f %f\n\n", moi[0][2], moi[1][2], moi[2][2]);
  
  cudaFree(dev_moi);
  cudaFree(dev_mass);
  cudaFree(dev_cm);
}
