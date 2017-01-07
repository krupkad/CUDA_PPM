#include "ppm.hpp"
#include "util/error.hpp"

#include <glm/gtc/type_ptr.hpp>

#include <iterator>
#include <algorithm>

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
  float tr = moi[0][0] + moi[1][1] + moi[2][2];
  for (int i = 0; i < 3; i++) {
  for (int j = 0; j < 3; j++) {
    moi[i][j] = ((i == j) ? tr : 0.0f) - moi[i][j] + mass*cm[i]*cm[j];
  }}

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

__global__ void kMeshIntersect(bool exec, bool biDir,
                                int nSubFace, const int *vTessIdx, const float *vTessData,
                                const glm::vec3 p0, const glm::vec3 dir,
                                unsigned int *count, float2 *uvOut, float *tOut) {
  int fSubIdx = threadIdx.x + blockIdx.x * blockDim.x;
  if (fSubIdx >= nSubFace)
    return;
  
  vTessIdx = &vTessIdx[3*fSubIdx];
  const float *v0 = &vTessData[PPM_NVARS*vTessIdx[0]];
  const float *v1 = &vTessData[PPM_NVARS*vTessIdx[1]];
  const float *v2 = &vTessData[PPM_NVARS*vTessIdx[2]];
  
  float e1[3], e2[3];
  for (int i = 0; i < 3; i++) {
    e1[i] = v1[i] - v0[i];
    e2[i] = v2[i] - v0[i];
  }

  float p[3];
  p[0] = dir[1]*e2[2] - dir[2]*e2[1];
  p[1] = dir[2]*e2[0] - dir[0]*e2[2];
  p[2] = dir[0]*e2[1] - dir[1]*e2[0];
  
  float idet = e1[0]*p[0] + e1[1]*p[1] + e1[2]*p[2];
  if (idet > -1e-5 && idet < 1e-5)
    return;
  idet = 1.0f/idet;
  
  float T[3];
  for (int i = 0; i < 3; i++)
    T[i] = p0[i] - v0[i];
  float u = idet*(p[0]*T[0] + p[1]*T[1] + p[2]*T[2]);
  if (u < 0 || u > 1)
    return;
  
  
  p[0] = T[1]*e1[2] - T[2]*e1[1];
  p[1] = T[2]*e1[0] - T[0]*e1[2];
  p[2] = T[0]*e1[1] - T[1]*e1[0];
  float v = idet*(dir[0]*p[0] + dir[1]*p[1] + dir[2]*p[2]);
  if (v < 0 || u+v > 1)
    return;
  
  float t = idet*(e2[0]*p[0] + e2[1]*p[1] + e2[2]*p[2]);
  if (biDir || (t > 1e-5)) {
    if (exec) {
      int oIdx = atomicSub(count, 1) - 1;
      uvOut[oIdx].x = u;
      uvOut[oIdx].y = v;
      tOut[oIdx] = t;
    } else {
      printf("%d %f\n", fSubIdx, t);
      atomicAdd(count, 1);
    }
  }
}

bool PPM::intersect(const glm::vec3 &p0, const glm::vec3 &dir, float2 &uv) {
  if (!isBuilt)
    return false;

  unsigned int *dev_count;
  cudaMalloc(&dev_count, sizeof(unsigned int));
  cudaMemset(dev_count, 0, sizeof(unsigned int));
  dim3 blkCnt((nFace*nSubFace + 1023) / 1024), blkDim(1024);
  kMeshIntersect<<<blkCnt,blkDim>>>(false, false, nFace*nSubFace, dev_tessIdx, dev_tessVtx, p0, dir, dev_count, nullptr, nullptr);
  checkCUDAError("kMeshIntersect", __LINE__);
  
  unsigned int count;
  cudaMemcpy(&count, dev_count, sizeof(unsigned int), cudaMemcpyDeviceToHost);
  if (!count) {
    printf("no ix\n");
    cudaFree(dev_count);
    return false;
  }
  
  float2 *dev_uvOut;
  float *dev_tOut;
  cudaMalloc(&dev_uvOut, count*sizeof(float2));
  cudaMalloc(&dev_tOut, count*sizeof(float));
  kMeshIntersect<<<blkCnt,blkDim>>>(true, false, nFace*nSubFace, dev_tessIdx, dev_tessVtx, p0, dir, dev_count, dev_uvOut, dev_tOut);
  checkCUDAError("kMeshIntersect", __LINE__);
  
  std::vector<float2> uvOut(count);
  std::vector<float> tOut(count);
  cudaMemcpy(&uvOut[0], dev_uvOut, count*sizeof(float2), cudaMemcpyDeviceToHost);
  cudaMemcpy(&tOut[0], dev_uvOut, count*sizeof(float), cudaMemcpyDeviceToHost);
  cudaFree(dev_uvOut);
  cudaFree(dev_tOut);
  
  int minPos = std::min_element(tOut.begin(), tOut.end()) - tOut.begin();
  uv = uvOut[minPos];
  
  printf("ix %f %f\n", uv.x, uv.y);
  return true;
}
