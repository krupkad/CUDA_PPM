#include "ppm.hpp"
#include "bezier.hpp"
#include "util/error.hpp"

#include <glm/gtc/type_ptr.hpp>

#include <iterator>
#include <algorithm>

__global__ void kCalcInertia(int nFace, const HeData *heFaces, const float *vtxData, glm::mat3 *moiOut, float *massOut, glm::vec3 *cmOut) {
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
      pA[3*i + j] = vtxData[PPM_NVARS*heFaces[3*fIdx + i].src + j];
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

__global__ void kCenter(int nVtx, glm::vec3 cm, float *vData) {
  int vIdx = threadIdx.x + blockDim.x * blockIdx.x;
  if (vIdx >= nVtx)
    return;
  
  vData[PPM_NVARS*vIdx + 0] -= cm[0];
  vData[PPM_NVARS*vIdx + 1] -= cm[1];
  vData[PPM_NVARS*vIdx + 2] -= cm[2];
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
  checkCUDAError("kCalcInertia", __LINE__);

  fprintf(stderr, "phys copy\n");
  cudaMemcpy(&moi, dev_moi, sizeof(glm::mat3), cudaMemcpyDeviceToHost);
  cudaMemcpy(&cm, dev_cm, sizeof(glm::vec3), cudaMemcpyDeviceToHost);
  cudaMemcpy(&mass, dev_mass, sizeof(float), cudaMemcpyDeviceToHost);

  fprintf(stderr, "phys recenter\n");
  kCenter<<<blkCnt,blkDim>>>(nVtx, cm, dev_vList);
  checkCUDAError("kCenter", __LINE__);
  cudaMemcpy(&vList[0], dev_vList, PPM_NVARS*nVtx*sizeof(float), cudaMemcpyDeviceToHost);
  
  fprintf(stderr, "phys reduce\n");
  float tr = moi[0][0] + moi[1][1] + moi[2][2];
  for (int i = 0; i < 3; i++) {
  for (int j = 0; j < 3; j++) {
    moi[i][j] = ((i == j) ? tr : 0.0f) - moi[i][j];
  }}
  mass *= 1000.0f;

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
                                unsigned int *count, int *idxOut, float *tOut) {
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
      tOut[oIdx] = t;
      idxOut[oIdx] = fSubIdx;
    } else {
      atomicAdd(count, 1);
    }
  }
}

__global__ void kUpdateCoeff(int nBasis2, int nVtx, const float *V, float sigma, const float *dv, float *coeff, float dt) {
  int bIdx = threadIdx.x + blockIdx.x * blockDim.x;
  int vIdx = threadIdx.y + blockIdx.y * blockDim.y;
  if (vIdx >= nVtx || bIdx >= nBasis2)
    return;

  dv = &dv[9*vIdx];

  float v = sigma * V[0*nBasis2 + bIdx];
  int tIdx = bIdx + vIdx*nBasis2;
  coeff[tIdx + 0 * nVtx*nBasis2] += dv[3] * v * dt;
  coeff[tIdx + 1 * nVtx*nBasis2] += dv[4] * v * dt;
  coeff[tIdx + 2 * nVtx*nBasis2] += dv[5] * v * dt;
}

__global__ void kPhysVerlet1(int nVtx, float *dv, float mass, const int2 *vBndList, float dt) {
  int vIdx = threadIdx.x + blockIdx.x * blockDim.x;
  if (vIdx >= nVtx)
    return;

  const int2 &bnd = vBndList[vIdx];
  mass *= (bnd.y - bnd.x);

  dv = &dv[9*vIdx];
  dv[3] += 0.5f*dv[6]*dt / mass;
  dv[4] += 0.5f*dv[7]*dt / mass;
  dv[5] += 0.5f*dv[8]*dt / mass;
  dv[0] += dv[3]*dt;
  dv[1] += dv[4]*dt;
  dv[2] += dv[5]*dt;

  dv[6] = 0.0f;
  dv[7] = 0.0f;
  dv[8] = 0.0f;
}

__global__ void kPhysNeighbor(int nVtx, const HeData *heLoops, const int2 *vBndList,
                                 float kSelf, float kDamp, float kNbr, float *dv) {
  int vIdx = threadIdx.x + blockIdx.x * blockDim.x;
  if (vIdx >= nVtx)
    return;
  const int2 &bnd = vBndList[vIdx];

  extern __shared__ float physSM[];
  float *vSM = &physSM[3 * threadIdx.x];
  kSelf = -(kSelf + kNbr*(bnd.y - bnd.x));
  vSM[0] = kSelf * dv[9*vIdx + 0];
  vSM[1] = kSelf * dv[9*vIdx + 1];
  vSM[2] = kSelf * dv[9*vIdx + 2];

  for (int i = bnd.x; i < bnd.y; i++) {
    int tgt = heLoops[i].tgt;
    vSM[0] += kNbr * dv[9*tgt + 0];
    vSM[1] += kNbr * dv[9*tgt + 1];
    vSM[2] += kNbr * dv[9*tgt + 2];
  }

  dv[9*vIdx + 6] = vSM[0] - kDamp*dv[9*vIdx + 3];
  dv[9*vIdx + 7] = vSM[1] - kDamp*dv[9*vIdx + 4];
  dv[9*vIdx + 8] = vSM[2] - kDamp*dv[9*vIdx + 5];
}

__global__ void kPhysVerlet2(int nVtx, float *dv, float dt) {
  int vIdx = threadIdx.x + blockIdx.x * blockDim.x;
  if (vIdx >= nVtx)
    return;

  dv = &dv[9*vIdx];
  dv[3] += 0.5f*dv[6]*dt;
  dv[4] += 0.5f*dv[7]*dt;
  dv[5] += 0.5f*dv[8]*dt;
}


/*
  
  if (uv.x > 0 && uv.y > 0) {
    fSubIdx = fIdx*nSubFace + UV_IDX(uv.x - 1, uv.y - 1) + nSubVtx - nSub - 1;
    idxOut[3*fSubIdx + 0] = tessGetIdx(uv.x, uv.y, heLoops, heFaces,  heTessOrder, fIdx, nVtx, nHe, nFace, nSub);
    idxOut[3*fSubIdx + 1] = tessGetIdx(uv.x-1, uv.y, heLoops, heFaces,  heTessOrder, fIdx, nVtx, nHe, nFace, nSub);
    idxOut[3*fSubIdx + 2] = tessGetIdx(uv.x, uv.y-1, heLoops, heFaces,  heTessOrder, fIdx, nVtx, nHe, nFace, nSub);
  }

  if (uv.x+uv.y < nSub) {
    fSubIdx = fIdx*nSubFace + UV_IDX(uv.x, uv.y);
    idxOut[3*fSubIdx + 0] = tessGetIdx(uv.x, uv.y, heLoops, heFaces,  heTessOrder, fIdx, nVtx, nHe, nFace, nSub);
    idxOut[3*fSubIdx + 1] = tessGetIdx(uv.x+1, uv.y, heLoops, heFaces,  heTessOrder, fIdx, nVtx, nHe, nFace, nSub);
    idxOut[3*fSubIdx + 2] = tessGetIdx(uv.x, uv.y + 1, heLoops, heFaces, heTessOrder, fIdx, nVtx, nHe, nFace, nSub);
  }
*/
__global__ void kPhysClick(int nSub, int nSubVtx, int fSubIdx, const float2 *uvIdx, 
                            const HeData *heFaces, const float *vData,
                            float force, float *dv) {
  int fIdx = fSubIdx / (nSub*nSub);
  int uvOff = fSubIdx - fIdx*nSub*nSub;

  float2 uv;
  if (uvOff >= nSubVtx - nSub - 1) {
    uv = uvIdx[uvOff - (nSubVtx - nSub - 1)];
    uv.x += 1.0f/nSub;
    uv.y += 1.0f/nSub;
  } else {
    uv = uvIdx[uvOff];
  }
  float w = 1.0f - uv.x - uv.y;

  const HeData &he0 = heFaces[3*fIdx], &he1 = heFaces[3*fIdx+1], &he2 = heFaces[3*fIdx+2];
  const float *v0 = &vData[PPM_NVARS*he0.src], *v1 = &vData[PPM_NVARS*he1.src], *v2 = &vData[PPM_NVARS*he2.src];
  float *dv0 = &dv[9*he0.src], *dv1 = &dv[9*he1.src], *dv2 = &dv[9*he2.src];

  float dx = force;
  dv0[6] += w * dx * v0[3];
  dv0[7] += w * dx * v0[4];
  dv0[8] += w * dx * v0[5];
  dv1[6] += uv.x * dx * v1[3];
  dv1[7] += uv.x * dx * v1[4];
  dv1[8] += uv.x * dx * v1[5];
  dv2[6] += uv.y * dx * v2[3];
  dv2[7] += uv.y * dx * v2[4];
  dv2[8] += uv.y * dx * v2[5];
}

int PPM::intersect(const glm::vec3 &p0, const glm::vec3 &dir) {
  if (!isBuilt)
    return false;

  unsigned int *dev_count;
  cudaMalloc(&dev_count, sizeof(unsigned int));
  cudaMemset(dev_count, 0, sizeof(unsigned int));
  dim3 blkCnt((nFace*nSubFace + 255) / 256), blkDim(256);
  kMeshIntersect<<<blkCnt,blkDim>>>(false, false, nFace*nSubFace, dev_tessIdx, dev_tessVtx, p0, dir, dev_count, nullptr, nullptr);
  checkCUDAError("kMeshIntersect", __LINE__);
  
  unsigned int count;
  cudaMemcpy(&count, dev_count, sizeof(unsigned int), cudaMemcpyDeviceToHost);
  if (!count) {
    //printf("no ix\n");
    cudaFree(dev_count);
    return -1;
  }
  
  float2 *dev_uvOut;
  float *dev_tOut;
  int *dev_idxOut;
  cudaMalloc(&dev_tOut, count*sizeof(float));
  cudaMalloc(&dev_idxOut, count*sizeof(int));
  kMeshIntersect<<<blkCnt,blkDim>>>(true, false, nFace*nSubFace, dev_tessIdx, dev_tessVtx, p0, dir, dev_count, dev_idxOut, dev_tOut);
  checkCUDAError("kMeshIntersect", __LINE__);
  
  std::vector<float> tOut(count);
  std::vector<int> idxOut(count);
  cudaMemcpy(&tOut[0], dev_tOut, count*sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(&idxOut[0], dev_idxOut, count*sizeof(int), cudaMemcpyDeviceToHost);
  
  int minPos = std::min_element(tOut.begin(), tOut.end()) - tOut.begin();
  int idx = idxOut[minPos];

  //printf("ix %f %f\n", uv.x, uv.y);
  cudaFree(dev_tOut);
  cudaFree(dev_idxOut);
  return idx;
}

void PPM::updateCoeff(int clickIdx, float clickForce,  float dt) {
  dim3 blkDim(128), blkCnt;
  blkCnt.x = (nVtx + blkDim.x - 1) / blkDim.x;
  kPhysVerlet1<<<blkCnt,blkDim>>>(nVtx, dev_dv, mass/nHe, dev_vBndList, dt);
  checkCUDAError("kPhysVerlet1", __LINE__); 
  
  blkDim.x = 16;
  blkDim.y = 64;
  blkCnt.x = (nBasis2 + blkDim.x - 1) / blkDim.x;
  blkCnt.y = (nVtx + blkDim.y - 1) / blkDim.y;
  kUpdateCoeff<<<blkCnt,blkDim>>>(nBasis2, nVtx, bezier->dev_V, 1.0, dev_dv, dev_coeff, dt);
  checkCUDAError("kUpdateCoeff", __LINE__);

  blkDim.x = 256;
  blkDim.y = 1;
  blkCnt.x = (nVtx + blkDim.x - 1) / blkDim.x;
  blkCnt.y = 1;
  int nSM = 3*blkDim.x*sizeof(float);
  kPhysNeighbor<<<blkCnt,blkDim,nSM>>>(nVtx, dev_heLoops, dev_vBndList, kSelf, kDamp, kNbr, dev_dv);
  if (clickIdx > -1)
    kPhysClick<<<1,1>>>(nSub, nSubVtx, clickIdx, dev_uvIdxMap, dev_heFaces, dev_vList, 1000.0f*clickForce, dev_dv);
  checkCUDAError("kPhysNeighbor", __LINE__);
  
  blkCnt.x = (nVtx + blkDim.x - 1) / blkDim.x;
  kPhysVerlet2<<<blkCnt,blkDim>>>(nVtx, dev_dv, dt);
  checkCUDAError("kPhysVerlet2", __LINE__); 
}

