#include "ppm.hpp"
#include "bezier.hpp"
#include "util/error.hpp"
#include "util/uvidx.hpp"

#include <iterator>
#include <algorithm>

__global__ void kCalcInertia(int nFace, const HeData *heFaces, const float
    *vtxData, float *moiOut, float *volOut, float *cmOut) {
  int fIdx = threadIdx.x + blockIdx.x * blockDim.x;
  if (fIdx >= nFace)
    return;

  extern __shared__ float matSM[];
  float *pA = &matSM[12*threadIdx.x];
  float *r = &matSM[12*threadIdx.x + 9];

  for (int i = 0; i < 3; i++) {
  for (int j = 0; j < 3; j++) {
    pA[3*i + j] = vtxData[PPM_NVARS*heFaces[3*fIdx + i].src + j];
  }}
  r[0] = pA[0] + pA[3] + pA[6];
  r[1] = pA[1] + pA[4] + pA[7];
  r[2] = pA[2] + pA[5] + pA[8];

  float detA = pA[0]*(pA[4]*pA[8]-pA[5]*pA[7])
              - pA[3]*(pA[1]*pA[8]-pA[7]*pA[2])
              + pA[6]*(pA[1]*pA[5]-pA[2]*pA[4]);
  atomicAdd(&cmOut[0], detA*r[0]/18.0f);
  atomicAdd(&cmOut[1], detA*r[1]/18.0f);
  atomicAdd(&cmOut[2], detA*r[2]/18.0f);
  atomicAdd(volOut, detA/6.0f);

  detA *= 60.0f*nFace;
  atomicAdd(&moiOut[0], (r[0]*r[0] + pA[0]*pA[0] + pA[3]*pA[3] + pA[6]*pA[6])/detA);
  atomicAdd(&moiOut[1], (r[0]*r[1] + pA[0]*pA[1] + pA[3]*pA[4] + pA[6]*pA[7])/detA);
  atomicAdd(&moiOut[2], (r[0]*r[2] + pA[2]*pA[0] + pA[5]*pA[3] + pA[8]*pA[6])/detA);
  atomicAdd(&moiOut[3], (r[1]*r[0] + pA[0]*pA[1] + pA[3]*pA[4] + pA[6]*pA[7])/detA);
  atomicAdd(&moiOut[4], (r[1]*r[1] + pA[1]*pA[1] + pA[4]*pA[4] + pA[7]*pA[7])/detA);
  atomicAdd(&moiOut[5], (r[1]*r[2] + pA[1]*pA[2] + pA[4]*pA[5] + pA[7]*pA[8])/detA);
  atomicAdd(&moiOut[6], (r[2]*r[0] + pA[2]*pA[0] + pA[5]*pA[3] + pA[8]*pA[6])/detA);
  atomicAdd(&moiOut[7], (r[2]*r[1] + pA[1]*pA[2] + pA[4]*pA[5] + pA[7]*pA[8])/detA);
  atomicAdd(&moiOut[8], (r[2]*r[2] + pA[2]*pA[2] + pA[5]*pA[5] + pA[8]*pA[8])/detA);
}

void PPM::physCalc() {
  cudaMemset(dev_moi, 0, 9*sizeof(float));
  cudaMemset(dev_cm, 0, 3*sizeof(float));
  cudaMemset(dev_vol, 0, sizeof(float));

  fprintf(stderr, "phys compute\n");
  dim3 blkDim(256), blkCnt((nFace + 255)/256);
  int nSM = (blkDim.x) * 12 * sizeof(float);
  kCalcInertia<<<blkCnt,blkDim,nSM>>>(nFace, dev_heFaces, dev_vList, dev_moi, dev_vol, dev_cm);
  checkCUDAError("kCalcInertia", __LINE__);

  fprintf(stderr, "phys copy\n");
  cudaMemcpy(moi, dev_moi, 9*sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(cm, dev_cm, 3*sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(&vol, dev_vol, sizeof(float), cudaMemcpyDeviceToHost);

  fprintf(stderr, "phys moi finalize\n");
  float tr = moi[0] + moi[4] + moi[8];
  for (int i = 0; i < 3; i++) {
  for (int j = 0; j < 3; j++) {
    moi[3*i+j] = ((i == j) ? tr : 0.0f) - moi[3*i+j];
  }}

  fprintf(stderr, "phys result: %f (%f %f %f)\n", vol, cm[0], cm[1], cm[2]);
  fprintf(stderr, "%f %f %f\n", moi[0], moi[3], moi[6]);
  fprintf(stderr, "%f %f %f\n", moi[1], moi[4], moi[7]);
  fprintf(stderr, "%f %f %f\n\n", moi[2], moi[5], moi[8]);
}

void PPM::physInit() {
  fprintf(stderr, "phys alloc\n");
  devAlloc(&dev_moi, 9*sizeof(float));
  devAlloc(&dev_cm, 3*sizeof(float));
  devAlloc(&dev_vol, sizeof(float));
  physCalc();
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

__global__ void kPhysVerlet2(int nVtx, float *dv, float mass, const int2 *vBndList, float dt)  {
  int vIdx = threadIdx.x + blockIdx.x * blockDim.x;
  if (vIdx >= nVtx)
    return;

  const int2 &bnd = vBndList[vIdx];
  mass *= (bnd.y - bnd.x);

  dv = &dv[9*vIdx];
  dv[3] += 0.5f*dv[6]*dt/mass;
  dv[4] += 0.5f*dv[7]*dt/mass;
  dv[5] += 0.5f*dv[8]*dt/mass;
}

__global__ void kPhysClick(int nSub, int nSubVtx, int fSubIdx,
                            const HeData *heFaces, const float *vData, const float *force,
                            float *dv) {
  int fIdx = fSubIdx / (nSub*nSub);
  int uvOff = fSubIdx - fIdx*nSub*nSub;

  float2 uv;
  if (uvOff >= nSubVtx - nSub - 1) {
    fUvInternalIdxMap(uvOff-(nSubVtx-nSub-1), uv, nSub);
    uv.x += 1.0f/nSub;
    uv.y += 1.0f/nSub;
  } else {
    fUvInternalIdxMap(uvOff, uv, nSub);
  }
  float w = 1.0f - uv.x - uv.y;

  const HeData &he0 = heFaces[3*fIdx], &he1 = heFaces[3*fIdx+1], &he2 = heFaces[3*fIdx+2];
  const float *v0 = &vData[PPM_NVARS*he0.src], *v1 = &vData[PPM_NVARS*he1.src], *v2 = &vData[PPM_NVARS*he2.src];
  float *dv0 = &dv[9*he0.src], *dv1 = &dv[9*he1.src], *dv2 = &dv[9*he2.src];

  dv0[6] += w * force[0] ;
  dv0[7] += w * force[1] ;
  dv0[8] += w * force[2] ;
  dv1[6] += uv.x * force[0] ;
  dv1[7] += uv.x * force[1] ;
  dv1[8] += uv.x * force[2] ;
  dv2[6] += uv.y * force[0] ;
  dv2[7] += uv.y * force[1] ;
  dv2[8] += uv.y * force[2] ;
}

__global__ void kPhysNonInertial(int nVtx, glm::vec3 angVel, const float *cm,
                                 const float *vTessData, float *dv) {
  int vIdx = threadIdx.x + blockIdx.x * blockDim.x;
  if (vIdx >= nVtx)
    return;

  vTessData = &vTessData[PPM_NVARS*vIdx];
  dv = &dv[9*vIdx];

  glm::vec3 r(vTessData[0] - cm[0], vTessData[1] - cm[1], vTessData[2] - cm[2]);
  float dot = r[0]*angVel[0] + r[1]*angVel[1] + r[2]*angVel[2];
  float ang = angVel[0]*angVel[0] + angVel[1]*angVel[1] + angVel[2]*angVel[2];

  glm::vec3 cf;
  cf[0] = dot*angVel[0] - ang*r[0];
  cf[1] = dot*angVel[1] - ang*r[1];
  cf[2] = dot*angVel[2] - ang*r[2];
  dv[6] += cf[0];
  dv[7] += cf[1];
  dv[8] += cf[2];
}

__global__ void kPhysClick_RigidBody(int nSub, int nSubVtx, int fSubIdx,
                            const HeData *heFaces, const float *cm, const float *vData,
                           const float *force, float *rbTorque) {
  int fIdx = fSubIdx / (nSub*nSub);
  int uvOff = fSubIdx - fIdx*nSub*nSub;

  float2 uv;
  if (uvOff >= nSubVtx - nSub - 1) {
    fUvInternalIdxMap(uvOff-(nSubVtx-nSub-1), uv, nSub);
    uv.x += 1.0f/nSub;
    uv.y += 1.0f/nSub;
  } else {
    fUvInternalIdxMap(uvOff, uv, nSub);
  }
  float w = 1.0f - uv.x - uv.y;

  const HeData &he0 = heFaces[3*fIdx], &he1 = heFaces[3*fIdx+1], &he2 = heFaces[3*fIdx+2];
  const float *v0 = &vData[PPM_NVARS*he0.src], *v1 = &vData[PPM_NVARS*he1.src], *v2 = &vData[PPM_NVARS*he2.src];

  glm::vec3 arm;
  for (int i = 0; i < 3; i++)
    arm[i] = w*(v0[i]-cm[i]) + uv.x*(v1[i]-cm[i]) + uv.y*(v2[i]-cm[i]);
  rbTorque[0] = arm[1]*force[2] - arm[2]*force[1];
  rbTorque[1] = arm[2]*force[0] - arm[0]*force[2];
  rbTorque[2] = arm[0]*force[1] - arm[1]*force[0];
}

int PPM::intersect(const glm::vec3 &p0, const glm::vec3 &dir) {
  if (!isBuilt)
    return false;

  glm::mat4 im = glm::inverse(model) ;
  glm::vec4 lp0_4(0.0f), ldir_4(0.0f);
  printf("%f %f %f\n", p0.x, p0.y, p0.z);
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 3; j++) {
      lp0_4[i] += im[j][i]*p0[j];
      ldir_4[i] += im[j][i]*dir[j];
    }
    lp0_4[i] += im[3][i];
  }
  glm::vec3 lp0(lp0_4.x/lp0_4.w, lp0_4.y/lp0_4.w, lp0_4.z/lp0_4.w);
  glm::vec3 ldir(ldir_4.x, ldir_4.y, ldir_4.z);

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

  float *dev_tOut;
  int *dev_idxOut;
  cudaMalloc(&dev_tOut, count*sizeof(float));
  cudaMalloc(&dev_idxOut, count*sizeof(int));
  kMeshIntersect<<<blkCnt,blkDim>>>(true, false, nFace*nSubFace, dev_tessIdx, dev_tessVtx, lp0, ldir, dev_count, dev_idxOut, dev_tOut);
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

void PPM::updateSb(float dt) {
  dim3 blkDim, blkCnt;

  // softbody VV - velocity half-update
  blkDim.x = 128;
  blkCnt.x = (nVtx + blkDim.x - 1) / blkDim.x;
  kPhysVerlet1<<<blkCnt,blkDim>>>(nVtx, dev_dv, mass/nHe, dev_vBndList, dt);
  checkCUDAError("kPhysVerlet1", __LINE__);

  // softbody VV - position update
  blkDim.x = 16;
  blkDim.y = 64;
  blkCnt.x = (nBasis2 + blkDim.x - 1) / blkDim.x;
  blkCnt.y = (nVtx + blkDim.y - 1) / blkDim.y;
  kUpdateCoeff<<<blkCnt,blkDim>>>(nBasis2, nVtx, bezier->dev_V, 1.0, dev_dv, dev_coeff, dt);
  checkCUDAError("kUpdateCoeff", __LINE__);
  physCalc();

  // softbody VV - force update
  blkDim.x = 256;
  blkDim.y = 1;
  blkCnt.x = (nVtx + blkDim.x - 1) / blkDim.x;
  blkCnt.y = 1;
  int nSM = 3*blkDim.x*sizeof(float);
  kPhysNeighbor<<<blkCnt,blkDim,nSM>>>(nVtx, dev_heLoops, dev_vBndList, kSelf, kDamp, kNbr, dev_dv);
  checkCUDAError("kPhysNeighbor", __LINE__);
  kPhysNonInertial<<<blkCnt,blkDim>>>(nVtx, rbAngVel, dev_cm, dev_vList, dev_dv);
  checkCUDAError("kPhysNonInertial", __LINE__);

  // softbody VV - velocity half-update
  blkDim.x = 128;
  blkCnt.x = (nVtx + blkDim.x - 1) / blkDim.x;
  kPhysVerlet2<<<blkCnt,blkDim>>>(nVtx, dev_dv, mass/nHe, dev_vBndList, dt);
  checkCUDAError("kPhysVerlet2", __LINE__);
}

