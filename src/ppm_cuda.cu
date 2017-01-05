#include "ppm.hpp"
#include "bezier.hpp"

#include <thrust/iterator/counting_iterator.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/copy.h>

// output weights and bezier basis coefficients for tessellated subvertices of a patch
__global__ void kBezEval(int deg, int nBasis, int nSubVtx, const float2 *uvIdxMap, float *bzOut, float *wgtOut) {
  int dIdx = blockIdx.x * blockDim.x + threadIdx.x;
  int uvIdx = blockIdx.y * blockDim.y + threadIdx.y;
  if (dIdx >= deg || uvIdx >= nSubVtx)
    return;

  // each thread needs nBasis*sizeof(float2) to compute all basis functions
  // plus 2*sizeof(float2) to store p and (1-p)
  extern __shared__ float2 sAll[];
  int nSM = (nBasis + 2);
  int tIdx = threadIdx.x + blockDim.x * threadIdx.y;
  float2 &p = sAll[tIdx*nSM + 0];
  float2 &np = sAll[tIdx*nSM + 1];
  float2 *sWork = &sAll[tIdx*nSM + 2];

  // get sector-local xy
  const float2 &uv = uvIdxMap[uvIdx];
  float a = 2.0 / deg;
  float ca, sa;
  sincospif(a, &sa, &ca);
  float x = uv.x + uv.y*ca, y = uv.y*sa;
  a *= dIdx;
  sincospif(a, &sa, &ca);

  // calculate bernstein polynomials
  p = 0.5f*make_float2(x*ca - y*sa, x*sa + y*ca) + 0.5f;
  np = 1.0f - p;
  kBnBasis<float>(nBasis, p, np, sWork);

  // compute weight
  float h2 = cospif(1.0f / (deg > 4 ? deg : 4)), h1 = 0.25f*h2;
  float r = hypotf(x, y);
  float h = (r - h1) / (h2 - h1);
  float s = rsqrtf(1.0f - h) - rsqrtf(h);
  float w = (r < h1) ? 1.0f : ((r > h2) ? 0.0f :( 1.0f / (1.0f + expf(2.0f*s))));

  // tensor product and output
  int oIdx = dIdx*nSubVtx + uvIdx;
  for (int k = 0; k < nBasis; k++) {
  for (int j = 0; j < nBasis; j++) {
    bzOut[j + nBasis*k + oIdx*nBasis*nBasis] = sWork[j].x * sWork[k].y;
  }}
  wgtOut[oIdx] = w;
}

// given a sorted list, find indices of where each block starts/ends
__global__ void kGetLoopBoundaries(int nHe, const int4 *heList, int2 *vBndList) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= nHe)
    return;

  if (i == 0 || heList[i-1].x != heList[i].x)
    vBndList[heList[i].x].x = i;
  if (i == nHe - 1 || heList[i + 1].x != heList[i].x)
    vBndList[heList[i].x].y = i+1;
}

// given loop boundaries, fill in he.z (loop order) and he.w (loop degree)
__global__ void kGetHeRootInfo(int nHe, const int2 *vBndList, int4 *heList, int4 *heLoops) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= nHe)
    return;

  int4 &he = heList[i];
  const int2 &bnd = vBndList[he.x];
  he.w = bnd.y - bnd.x;
  he.z = -1;
  for (int j = bnd.x; j < bnd.y; j++) {
    if (heLoops[j].y == he.y) {
      he.z = j - bnd.x;
      return;
    }
  }
}

__global__ void kMeshSample(int nVert, int nGrid, int degMin,
                            cudaTextureObject_t sampTexObj,
                            const int4 *heLoops, const int2 *vHeLoopBnd,
                            const float *vData, float *samp) {
  int x = blockIdx.x * blockDim.x + threadIdx.x; // grid idx
  int y = blockIdx.y * blockDim.y + threadIdx.y; // grid idx
  int vtxIdx = blockIdx.z * blockDim.z + threadIdx.z; // vert idx
  if (x >= nGrid || y >= nGrid || vtxIdx >= nVert)
    return;

  const int2 &heBnd = vHeLoopBnd[vtxIdx];
  float4 uvi = tex2DLayered<float4>(sampTexObj, x, y, heBnd.y - heBnd.x - degMin);
  float u = uvi.x, v = uvi.y, w = 1.0f - u - v;
  int heOff = uvi.z;

  const int4 &he0 = heLoops[heBnd.x + heOff];
  const int4 &he1 = heLoops[heBnd.x + (heOff + 1)%he0.w];
  int sIdx = x + nGrid*y + vtxIdx*nGrid*nGrid, sDim = nGrid*nGrid*nVert;
  const float *p0, *p1, *p2;

  p0 = &vData[PPM_NVARS * he0.x];
  p1 = &vData[PPM_NVARS * he0.y];
  p2 = &vData[PPM_NVARS * he1.y];
  for (int i = 0; i < PPM_NVARS; i++)
    samp[sIdx + i * sDim] = p0[i] * w + p1[i] * u + p2[i] * v;
}

__global__ void kMeshSampleOrig(int nVert, int nGrid, int degMin,
                                cudaTextureObject_t sampTexObj,
                                const int4 *heLoops, const int2 *vHeLoopBnd,
                                const float *vtx, float *samp) {
  int ix = blockIdx.x * blockDim.x + threadIdx.x; // grid idx
  int iy = blockIdx.y * blockDim.y + threadIdx.y; // grid idx
  int vtxIdx = blockIdx.z * blockDim.z + threadIdx.z; // vert idx
  if (ix >= nGrid || iy >= nGrid || vtxIdx >= nVert)
    return;
  float x = 2.0f * float(ix) / (nGrid-1) - 1.0f;
  float y = 2.0f * float(iy) / (nGrid-1) - 1.0f;

  const int2 &heBnd = vHeLoopBnd[vtxIdx];
  int deg = heBnd.y - heBnd.x;
  float alpha = 2.0f*M_PI / deg;
  float th = ((y < 0) ? 2.0f*M_PI : 0.0f) + atan2f(y, x);
  float r = hypotf(x, y);
  float dTh = fmodf(th, alpha);
  int ord = floorf(th / alpha);
  float v = r*sinf(dTh) / sinf(alpha);
  float u = r*cosf(dTh) - v*cosf(alpha);
  float k = u + v;
  if (k > 1.0) {
	  u /= k;
	  v /= k;
  }
  float w = 1.0 - u - v;

  const int4 &he0 = heLoops[heBnd.x + ord];
  const int4 &he1 = heLoops[heBnd.x + (ord + 1) % he0.w];
  int sIdx = ix + nGrid*iy + vtxIdx*nGrid*nGrid, sDim = nGrid*nGrid*nVert;
  const float *p0, *p1, *p2;

  p0 = &vtx[PPM_NVARS * he0.x];
  p1 = &vtx[PPM_NVARS * he0.y];
  p2 = &vtx[PPM_NVARS * he1.y];
  for (int i = 0; i < PPM_NVARS; i++)
    samp[sIdx + i * sDim] = p0[i] * w + p1[i] * u + p2[i] * v;
}

__global__ void kGetHeTessInfo(int nHe, int degMin, const int4 *heLoops, int4 *heTarg, const int2 *vBndList) {
  int heIdx = threadIdx.x + blockIdx.x * blockDim.x;
  if (heIdx >= nHe)
    return;
  
  int4 &he = heTarg[heIdx];
  he.z = he.z + (he.w*(he.w - 1) - degMin*(degMin - 1)) / 2;
  
  he.w = -1;
  const int2 &bnd = vBndList[he.y];
  for (int i = bnd.x; i < bnd.y; i++) {
    if (heLoops[i].y == he.x) {
      he.w = i;
      break;
    }
  }
}

__global__ void kGetHeTessIdx(int nHe, const int4 *heLoops, int *heTessIdx) {
  int heIdx = threadIdx.x + blockIdx.x * blockDim.x;
  if (heIdx >= nHe)
    return;

  if (heIdx < heLoops[heIdx].w)
    heTessIdx[heIdx] = -heIdx-1;
  else
    heTessIdx[heIdx] = heLoops[heIdx].w+1;
}

__global__ void kGetHeTessOrder(int nHe, const int4 *dev_heLoops, const int *dev_heTessIdx, int4 *dev_heLoopsOrder) {
  int heIdx = threadIdx.x + blockIdx.x * blockDim.x;
  if (heIdx >= nHe/2)
    return;

  int idx = -dev_heTessIdx[heIdx]-1;
  const int4 &he0 = dev_heLoops[idx];
  const int4 &he1 = dev_heLoops[he0.w];
  dev_heLoopsOrder[heIdx] = make_int4(he0.x, he0.z, he1.x, he1.z);
}

__device__ inline float patchContrib(int dIdx, int vIdx, int nBasis2, const float *bez, const float *wgt, const float *coeff, float &res) {
  bez = &bez[dIdx*nBasis2];
  float w = wgt[dIdx];
  for (int i = 0; i < nBasis2; i++)
    res += w * bez[i] * coeff[i + vIdx * nBasis2];
  return w;
}

// generate the per-face template for tessellation
#define UV_IDX(u,v) (((u)+(v)+1)*((u)+(v))/2 + (v))
__global__ void kTessVtx_Face(int nVtx, int nFace, int nSub, int nBasis2,
                        const int4 *heFaces, const float *bezData, const float *wgtData, const float *coeff,
                        const int2 *uvIdxMap, float *vDataOut) {
  int fIdx = blockIdx.x * blockDim.x + threadIdx.x;
  int uvIdx = blockIdx.y * blockDim.y + threadIdx.y;
  int dataIdx = blockIdx.z * blockDim.z + threadIdx.z;
  int nSubVtx = (nSub+1)*(nSub+2)/2;
  if (fIdx >= nFace || uvIdx >= (nSub-1)*(nSub-2)/2 || dataIdx >= PPM_NVARS)
    return;
    
  const int2 &uv = uvIdxMap[uvIdx];
  const int4 &he0 = heFaces[3 * fIdx + 0];
  const int4 &he1 = heFaces[3 * fIdx + 1];
  const int4 &he2 = heFaces[3 * fIdx + 2];

  float res = 0.0, wgt = 0.0;
  wgt += patchContrib(he0.z*nSubVtx + UV_IDX(uv.x, uv.y), he0.x + dataIdx*nVtx, nBasis2, bezData, wgtData, coeff, res);
  wgt += patchContrib(he1.z*nSubVtx + UV_IDX(uv.y, nSub-uv.x-uv.y), he1.x + dataIdx*nVtx, nBasis2, bezData, wgtData, coeff, res);
  wgt += patchContrib(he2.z*nSubVtx + UV_IDX(nSub-uv.x-uv.y, uv.x), he2.x + dataIdx*nVtx, nBasis2, bezData, wgtData, coeff, res);

  vDataOut[PPM_NVARS*(fIdx*(nSub-2)*(nSub-1)/2 + UV_IDX(uv.x-1,uv.y-1)) + dataIdx] = res / wgt;
}

__global__ void kTessVtx_Edge(int nVtx, int nHe, int nSub, int nBasis2,
                        const int4 *heLoopsOrder,
                        const float *bezData, const float *wgtData, const float *coeff,
                        float *vDataOut) {
  int heIdx = blockIdx.x * blockDim.x + threadIdx.x;
  int uIdx = blockIdx.y * blockDim.y + threadIdx.y;
  int dataIdx = blockIdx.z * blockDim.z + threadIdx.z;
  if (heIdx >= nHe/2 || uIdx >= nSub-1 || dataIdx >= PPM_NVARS)
    return;

  const int4 &he = heLoopsOrder[heIdx];
  float res = 0.0, wgt = 0.0;
  wgt += patchContrib(he.y*(nSub+1)*(nSub+2)/2 + UV_IDX(uIdx+1,0), he.x + dataIdx*nVtx, nBasis2, bezData, wgtData, coeff, res);
  wgt += patchContrib(he.w*(nSub+1)*(nSub+2)/2 + UV_IDX(nSub-uIdx-1,0), he.z + dataIdx*nVtx, nBasis2, bezData, wgtData, coeff, res);

  vDataOut[PPM_NVARS*(heIdx*(nSub-1) + uIdx) + dataIdx] = res / wgt;
}


__global__ void kTessVtx_Vtx(int nVtx, int nSub,  int nBasis2,
                        const int4 *heLoops, const int2 *vBndList, 
                        const float *bezData, const float *wgtData, const float *coeff,
                        float *vDataOut) {
  int vIdx = blockIdx.x * blockDim.x + threadIdx.x;
  int dataIdx = blockIdx.y * blockDim.y + threadIdx.y;
  if (vIdx >= nVtx || dataIdx >= PPM_NVARS)
    return;

  const int4 &he = heLoops[vBndList[vIdx].x];
  int dIdx = he.z*(nSub+1)*(nSub+2)/2;
  
  const float *bez = &bezData[dIdx*nBasis2];
  float wgt = wgtData[dIdx], res;
  for (int i = 0; i < nBasis2; i++)
    res += wgt * bez[i] * coeff[i + (he.x + dataIdx * nVtx) * nBasis2];

  vDataOut[PPM_NVARS*vIdx + dataIdx] = res / wgt;
}

__global__ void kWeightScale(int nFace, int nSubVtx, float *vData, float *wgt) {
  int fIdx = blockIdx.x * blockDim.x + threadIdx.x;
  int uvIdx = blockIdx.y * blockDim.y + threadIdx.y;
  if (fIdx >= nFace || uvIdx >= nSubVtx)
    return;

  float w = wgt[fIdx*nSubVtx + uvIdx];

  for (int i = 0; i < PPM_NVARS; i++)
    vData[PPM_NVARS * (fIdx*nSubVtx + uvIdx) + i] /= w;
}

__global__ void kTessVtxSM(int nVtx, int nHe, int nSub, int nSubVtx, int nBasis2, int degMin,
  const int4 *heFaces, const float *bezData, const float *wgtData, const float *coeff,
  const int2 *uvIdxMap, float *vtxOut, float *wgtOut) {
  int heIdx = blockIdx.x * blockDim.x + threadIdx.x;
  int uvIdx = blockIdx.y * blockDim.y + threadIdx.y;
  int dataIdx = blockIdx.z * blockDim.z + threadIdx.z;
  if (heIdx >= nHe || dataIdx >= PPM_NVARS)
    return;

  const int4 &he = heFaces[heIdx];
  int dOff = he.z + (he.w*(he.w - 1) - degMin*(degMin - 1)) / 2;

  extern __shared__ float sTessVtxAltAll2[];
  float *sLoc = &sTessVtxAltAll2[(blockDim.x*threadIdx.z + threadIdx.x) * nBasis2];

  int fIdx = heIdx / 3;
  float *out = &vtxOut[PPM_NVARS * (fIdx*nSubVtx + uvIdx)];
  float *wgt = &wgtOut[fIdx*nSubVtx + uvIdx];

  for (int i = threadIdx.y; i < nBasis2; i += blockDim.y)
    sLoc[i] = coeff[i + he.x*nBasis2 + dataIdx*nBasis2*nVtx];
  __syncthreads();

  if (uvIdx >= nSubVtx)
    return;
  const int2 &uv = uvIdxMap[uvIdx];
  const int uvRot[4] = { uv.x, uv.y, nSub - uv.x - uv.y, uv.x };
  int uvIdxLoc = UV_IDX(uvRot[heIdx % 3], uvRot[heIdx % 3 + 1]);

  float v = 0.0, w = wgtData[dOff*nSubVtx + uvIdxLoc];
  for (int i = 0; i < nBasis2; i++)
    v += sLoc[i] * bezData[i + dOff*nSubVtx*nBasis2 + uvIdxLoc*nBasis2] * w;

  atomicAdd(&out[dataIdx], v);
  if (dataIdx == 0)
    atomicAdd(wgt, w);
}

__global__ void kGetNormals(int nHe, const int4 *heLoops, float *vData) {
  int heIdx = threadIdx.x + blockIdx.x * blockDim.x;
  if (heIdx >= nHe)
    return;

  const int4 &he0 = heLoops[heIdx];
  const int4 &he1 = heLoops[heIdx - he0.z + (he0.z + 1) % he0.w];
  float dx1[3], dx0[3];
  for (int i = 0; i < 3; i++) {
    dx1[i] = vData[PPM_NVARS * he1.y + i] - vData[PPM_NVARS * he1.x + i];
    dx0[i] = vData[PPM_NVARS * he0.y + i] - vData[PPM_NVARS * he0.x + i];
  }

  atomicAdd(&vData[PPM_NVARS * he0.x + 3], dx0[1] * dx1[2] - dx0[2] * dx1[1]);
  atomicAdd(&vData[PPM_NVARS * he0.x + 4], dx0[2] * dx1[0] - dx0[0] * dx1[2]);
  atomicAdd(&vData[PPM_NVARS * he0.x + 5], dx0[0] * dx1[1] - dx0[1] * dx1[0]);
}

__global__ void kUpdateCoeff(int nBasis2, int nSamp, float *V, float sigma, float *dv, float *coeff, float dt) {
  int bIdx = threadIdx.x + blockIdx.x * blockDim.x;
  int sIdx = threadIdx.y + blockIdx.y * blockDim.y;
  if (sIdx >= nSamp || bIdx >= nBasis2)
    return;

  float v = sigma * V[bIdx];
  int tIdx = bIdx + sIdx*nBasis2;
  coeff[tIdx + 0 * nSamp*nBasis2] += dv[6 * sIdx + 3] * v * dt;
  coeff[tIdx + 1 * nSamp*nBasis2] += dv[6 * sIdx + 4] * v * dt;
  coeff[tIdx + 2 * nSamp*nBasis2] += dv[6 * sIdx + 5] * v * dt;
  dv[6 * sIdx + 0] += dv[6 * sIdx + 3] * dt;
  dv[6 * sIdx + 1] += dv[6 * sIdx + 4] * dt;
  dv[6 * sIdx + 2] += dv[6 * sIdx + 5] * dt;
  dv[6 * sIdx + 3] -= 0.5f*dv[6 * sIdx + 0]*dt;
  dv[6 * sIdx + 4] -= 0.5f*dv[6 * sIdx + 1] * dt;
  dv[6 * sIdx + 5] -= 0.5f*dv[6 * sIdx + 2] * dt;
}

__global__ void kUpdateCoeffSM(int nBasis2, int nSamp, float *V, float sigma, float *dv, float *coeff, float dt) {
  int bIdx = threadIdx.x + blockIdx.x * blockDim.x;
  int sIdx = threadIdx.y + blockIdx.y * blockDim.y;
  if (sIdx >= nSamp || bIdx >= nBasis2)
    return;

  extern __shared__ float ucSM[];
  float *dvSM = &ucSM[6 * threadIdx.y];
  for (int i = threadIdx.x; i < 6; i += blockDim.x)
    dvSM[i] = dv[6 * sIdx + i];
  __syncthreads();


  float v = sigma * V[bIdx];
  int tIdx = bIdx + sIdx*nBasis2;
  coeff[tIdx + 0 * nSamp*nBasis2] += dvSM[0] * v * dt;
  coeff[tIdx + 1 * nSamp*nBasis2] += dvSM[1] * v * dt;
  coeff[tIdx + 2 * nSamp*nBasis2] += dvSM[2] * v * dt;
  dvSM[0] += dvSM[3] * dt;
  dvSM[1] += dvSM[4] * dt;
  dvSM[2] += dvSM[5] * dt;
  dvSM[3] -= 0.5f*dvSM[0] * dt;
  dvSM[4] -= 0.5f*dvSM[1] * dt;
  dvSM[5] -= 0.5f*dvSM[2] * dt;

  for (int i = threadIdx.x; i < 6; i += blockDim.x)
    dv[6 * sIdx + i] = dvSM[i];
  __syncthreads();
}

__device__ inline int tessGetIdx(int u, int v, const int4 *heLoops, const int4 *heFaces,
                                const int *heTessOrder,
                                int fIdx, int nVtx, int nHe, int nFace, int nSub) {
  const int4 &he0 = heFaces[3*fIdx+0];
  const int4 &he1 = heFaces[3*fIdx+1];
  const int4 &he2 = heFaces[3*fIdx+2];
  int heIdx1 = heTessOrder[he0.w];
  int heIdx2 = heTessOrder[he1.w];
  int heIdx3 = heTessOrder[he2.w];
  int w = nSub-u-v;
  
  if (u == 0 && v == 0)
    return nFace*(nSub-1)*(nSub-2)/2 + nHe*(nSub-1)/2 + he0.x;
  if (w == 0 && v == 0)
    return nFace*(nSub-1)*(nSub-2)/2 + nHe*(nSub-1)/2 + he1.x;
  if (u == 0 && w == 0)
    return nFace*(nSub-1)*(nSub-2)/2 + nHe*(nSub-1)/2 + he2.x;
  
  if (v == 0) {
    if (heIdx1 < nHe/2)
      return nFace*(nSub-1)*(nSub-2)/2 + heIdx1*(nSub-1) + w-1;
    else
      return nFace*(nSub-1)*(nSub-2)/2 + (nHe-heIdx1-1)*(nSub-1) + u-1;
  }
  if (w == 0) {
    if (heIdx2 < nHe/2)
      return nFace*(nSub-1)*(nSub-2)/2 + heIdx2*(nSub-1) + u-1;
    else
      return nFace*(nSub-1)*(nSub-2)/2 + (nHe-heIdx2-1)*(nSub-1) + v-1;
  }
  if (u == 0) {
    if (heIdx3 < nHe/2)
      return nFace*(nSub-1)*(nSub-2)/2 + heIdx3*(nSub-1) + v-1;
    else
      return nFace*(nSub-1)*(nSub-2)/2 + (nHe-heIdx3-1)*(nSub-1) + w-1;
  }
  
  return fIdx*(nSub-1)*(nSub-2)/2 + UV_IDX(u-1,v-1);
    
}

__global__ void kTessEdges(int nVtx, int nHe, int nFace, int nSub,
                                const int4 *heLoops, const int4 *heFaces, const int *heTessOrder,
                                const int2 *uvIdxMap, int *idxOut) {
  int fIdx = blockIdx.x * blockDim.x + threadIdx.x;
  int vSubIdx = blockIdx.y * blockDim.y + threadIdx.y;
  int nSubVtx = (nSub + 1)*(nSub + 2) / 2;
  int nSubFace = nSub*nSub;
  if (fIdx >= nFace || vSubIdx >= nSubVtx)
    return;

  int fSubIdx;
  const int2 &uv = uvIdxMap[vSubIdx];
  
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
}

// generate sampling pattern textures
void PPM::genSampTex() {
  fprintf(stderr, "populating patch maps (%d-%d = %d)\n", degMin, degMax, nDeg);
  float4 *sampTexData = new float4[nGrid * nGrid * nDeg];
  for (int d = 0; d < nDeg; d++) {
    for (int j = 0; j < nGrid; j++) {
      for (int i = 0; i < nGrid; i++) {
        float x(2 * i - nGrid + 1); x /= nGrid - 1;
        float y(2 * j - nGrid + 1); y /= nGrid - 1;

        float alpha = 2.0f*M_PI / (d + degMin);
        float th = ((y < 0) ? 2.0f*M_PI : 0.0f) + atan2f(y, x);
        float r = hypotf(x, y);
        float dTh = fmodf(th, alpha);
        int ord = floorf(th / alpha);
        float v = r*sinf(dTh) / sinf(alpha);
        float u = r*cosf(dTh) - v*cosf(alpha);
		    float w = 1.0f - u - v;
		    if (w < 0) {
			    float k = u + v;
			    u /= k;
			    v /= k;
			    w = 0.0f;
		    }

        sampTexData[i + j*nGrid + d*nGrid*nGrid] = make_float4(u, v, ord, 0);
      }
    }
  }

  fprintf(stderr, "allocating texture memory\n");
  dev_sampTexArray = nullptr;
  cudaChannelFormatDesc channel = cudaCreateChannelDesc<float4>();
  cudaMalloc3DArray(&dev_sampTexArray, &channel,
    make_cudaExtent(nGrid, nGrid, nDeg), cudaArrayLayered);
  checkCUDAError("cudaMalloc3DArray", __LINE__);

  cudaMemcpy3DParms cpyParms = { 0 };
  cpyParms.srcPos = make_cudaPos(0, 0, 0);
  cpyParms.dstPos = make_cudaPos(0, 0, 0);
  cpyParms.srcPtr = make_cudaPitchedPtr(sampTexData, nGrid*sizeof(float4), nGrid, nGrid);
  cpyParms.dstArray = dev_sampTexArray;
  cpyParms.extent = make_cudaExtent(nGrid, nGrid, nDeg);
  cpyParms.kind = cudaMemcpyHostToDevice;
  cudaMemcpy3D(&cpyParms);
  delete sampTexData;
  checkCUDAError("cudaMemcpy3D", __LINE__);

  fprintf(stderr, "creating texture\n");
  cudaResourceDesc resDesc;
  memset(&resDesc, 0, sizeof resDesc);
  resDesc.resType = cudaResourceTypeArray;
  resDesc.res.array.array = dev_sampTexArray;

  cudaTextureDesc texDesc;
  memset(&texDesc, 0, sizeof texDesc);
  texDesc.addressMode[0] = cudaAddressModeClamp;
  texDesc.addressMode[1] = cudaAddressModeClamp;
  texDesc.filterMode = cudaFilterModePoint;
  texDesc.readMode = cudaReadModeElementType;
  texDesc.normalizedCoords = 0;

  cudaCreateTextureObject(&sampTexObj, &resDesc, &texDesc, nullptr);
  //checkCUDAError("cudaCreateTextureObject", __LINE__);
}

void PPM::genCoeff() {
  dim3 blkDim;
  dim3 blkCnt;

  // generate mesh sample points
  blkDim.x = 4;
  blkDim.y = 4;
  blkDim.z = 32;
  blkCnt.x = (nGrid + blkDim.x - 1) / blkDim.x;
  blkCnt.y = (nGrid + blkDim.y - 1) / blkDim.y;
  blkCnt.z = (nVtx + blkDim.z - 1) / blkDim.z;
  if (canUseTexObjs && useSampTex) {
    kMeshSample<<<blkCnt, blkDim>>>(nVtx, nGrid, degMin,
      sampTexObj,
      dev_heLoops, dev_vBndList,
      dev_vList, dev_samp);
    checkCUDAError("kMeshSample", __LINE__);
  }
  else {
    kMeshSampleOrig<<<blkCnt, blkDim>>>(nVtx, nGrid, degMin,
      sampTexObj,
      dev_heLoops, dev_vBndList,
      dev_vList, dev_samp);
    checkCUDAError("kMeshSampleOrig", __LINE__);
  }

  bezier->getCoeff(nVtx, dev_samp, dev_coeff);
}

void PPM::devCoeffInit() {
   // allocate and generate coefficients
  devAlloc(&dev_samp, PPM_NVARS * nGrid2 * nVtx * sizeof(float));
  devAlloc(&dev_coeff, PPM_NVARS * nBasis2 * nVtx * sizeof(float));
  genCoeff();

  // initialize the deformation vector
  float *dv = new float[6 * nVtx];
  for (int i = 0; i < nVtx; i++) {
    dv[6 * i + 0] = 0.0f;
    dv[6 * i + 1] = 0.0f;
    dv[6 * i + 2] = 0.0f;
    dv[6 * i + 3] = 0.1 * (float(rand()) / RAND_MAX - 0.5f);
    dv[6 * i + 4] = 0.1 * (float(rand()) / RAND_MAX - 0.5f);
    dv[6 * i + 5] = 0.1 * (float(rand()) / RAND_MAX - 0.5f);
  }
  devAlloc(&dev_dv, 6 * nVtx*sizeof(float));
  cudaMemcpy(dev_dv, dv, 6 * nVtx*sizeof(float), cudaMemcpyHostToDevice);
  delete dv;
}

void PPM::devMeshInit() {
  fprintf(stderr, "uploading mesh data\n");
  devAlloc(&dev_vList, PPM_NVARS*nVtx*sizeof(float));
  cudaMemcpy(dev_vList, &vList[0],  PPM_NVARS*nVtx*sizeof(float), cudaMemcpyHostToDevice);
  devAlloc(&dev_heFaces, nHe*sizeof(int4));
  cudaMemcpy(dev_heFaces, &heFaces[0], nHe*sizeof(int4), cudaMemcpyHostToDevice);

  // populate the loops
  fprintf(stderr, "sorting loops\n");
  getHeLoops();
  devAlloc(&dev_heLoops, nHe*sizeof(int4));
  cudaMemcpy(dev_heLoops, &heLoops[0], nHe*sizeof(int4), cudaMemcpyHostToDevice);

  // fill in remaining halfedge data
  devAlloc(&dev_vBndList, nVtx*sizeof(int2));
  cudaMemset(dev_vBndList, 0xFF, nVtx*sizeof(int2));
  dim3 blkCnt((nHe + 1024 - 1) / 1024);
  dim3 blkDim(1024);
  kGetLoopBoundaries<<<blkCnt, blkDim>>>(nHe, dev_heLoops, dev_vBndList);
  kGetHeRootInfo<<<blkCnt, blkDim>>>(nHe, dev_vBndList, dev_heLoops, dev_heLoops);
  kGetHeRootInfo<<<blkCnt, blkDim>>>(nHe, dev_vBndList, dev_heFaces, dev_heLoops);

  // recalculate normals
  kGetNormals<<<blkCnt, blkDim>>>(nHe, dev_heLoops, dev_vList);
  checkCUDAError("kGetNormals", __LINE__);
}

void PPM::devPatchInit() {
  // initialize the bezier patch calculator
  bezier = new Bezier<float>(nBasis, nGrid);

  // build the uv index map
  fprintf(stderr, "creating uv index map %d\n", nSubVtx);
  float2 *uvIdxMap = new float2[nSubVtx];
  int2 *iuvIdxMap = new int2[nSubVtx];
  for (int v = 0; v <= nSub; v++) {
  for (int u = 0; u <= nSub - v; u++) {
	  uvIdxMap[UV_IDX(u, v)] = make_float2(float(u) / nSub, float(v) / nSub);
	  iuvIdxMap[UV_IDX(u, v)] = make_int2(u,v);
  }}
  devAlloc(&dev_uvIdxMap, nSubVtx*sizeof(float2));
  cudaMemcpy(dev_uvIdxMap, uvIdxMap, nSubVtx*sizeof(float2), cudaMemcpyHostToDevice);
  devAlloc(&dev_iuvIdxMap, nSubVtx*sizeof(int2));
  cudaMemcpy(dev_iuvIdxMap, iuvIdxMap, nSubVtx*sizeof(int2), cudaMemcpyHostToDevice);
  delete uvIdxMap;
  delete iuvIdxMap;

  if (nSub > 2) {
    int2 *iuvInternalIdxMap = new int2[(nSub-1)*(nSub-2)/2];
    for (int v = 0; v <= nSub-3; v++) {
    for (int u = 0; u <= nSub-3-v; u++) {
      iuvInternalIdxMap[UV_IDX(u,v)] = make_int2(u+1,v+1);
    }}
    devAlloc(&dev_iuvInternalIdxMap, (nSub-1)*(nSub-2)*sizeof(int2)/2);
    cudaMemcpy(dev_iuvInternalIdxMap, iuvInternalIdxMap, (nSub-1)*(nSub-2)*sizeof(int2)/2, cudaMemcpyHostToDevice);
    delete iuvInternalIdxMap;
  }

  // d*(d-1)/2 - dmin*(dmin-1)/2
  fprintf(stderr, "creating patch data\n");
  devAlloc(&dev_bezPatch, nBasis2*nSubVtx*((degMax + 1)*degMax / 2 - degMin*(degMin - 1) / 2)*sizeof(float));
  devAlloc(&dev_wgtPatch, nSubVtx*((degMax + 1)*degMax / 2 - degMin*(degMin - 1) / 2)*sizeof(float));
  dim3 blkSize(16,16), blkCnt;
  blkCnt.x = (nDeg+blkSize.x-1)/blkSize.x;
  blkCnt.y = (nSubVtx + blkSize.y - 1) / blkSize.y;
  int nTessSM = (nBasis + 2) * blkSize.x * blkSize.y * sizeof(float2);
  for (int d = degMin; d <= degMax; d++) {
    int dOff = d*(d - 1) - degMin*(degMin - 1);
    dOff /= 2;
    kBezEval<<<blkCnt,blkSize,nTessSM>>>(d, nBasis, nSubVtx, dev_uvIdxMap, &dev_bezPatch[dOff*nSubVtx*nBasis2], &dev_wgtPatch[dOff*nSubVtx]);
    checkCUDAError("kBezEval", __LINE__);
  }
  
  fprintf(stderr, "creating sample data\n");
  if (canUseTexObjs) {
    genSampTex();
    checkCUDAError("genSampTex", __LINE__);
  }
}

void PPM::devTessInit() {
  fprintf(stderr, "creating edge tesselation\n");
  if (useVisualize) {
    size_t nBytes;
    cudaGraphicsMapResources(1, &dev_vboTessIdx, 0);
    cudaGraphicsResourceGetMappedPointer((void**)&dev_tessIdx, &nBytes, dev_vboTessIdx);
  } else {
    devAlloc(&dev_tessIdx, 3*nFace*nSubFace*sizeof(int));
  }
  
  dim3 blkSize(1024), blkCnt((nHe +1023) / 1024);
  kGetHeTessInfo<<<blkCnt, blkSize>>>(nHe, degMin, dev_heLoops, dev_heFaces, dev_vBndList);
  kGetHeTessInfo<<<blkCnt, blkSize>>>(nHe, degMin, dev_heLoops, dev_heLoops, dev_vBndList);

  int *dev_heTessIdx;
  cudaMalloc(&dev_heTessIdx, nHe*sizeof(int));
  kGetHeTessIdx<<<blkCnt, blkSize>>>(nHe, dev_heLoops, dev_heTessIdx);

  thrust::counting_iterator<int> order_itr(0);
  thrust::device_vector<int> order_vec(order_itr, order_itr+nHe);
  thrust::device_ptr<int> heTessIdx_ptr(dev_heTessIdx);
  thrust::sort_by_key(heTessIdx_ptr, heTessIdx_ptr+nHe, order_vec.begin());

  devAlloc(&dev_heLoopsOrder, nHe*sizeof(int4)/2);
  kGetHeTessOrder<<<blkCnt, blkSize>>>(nHe, dev_heLoops, dev_heTessIdx, dev_heLoopsOrder);
  thrust::copy(order_itr, order_itr+nHe, heTessIdx_ptr);
  thrust::sort_by_key(order_vec.begin(), order_vec.end(), heTessIdx_ptr);
  
  blkSize.x = 64;
  blkSize.y = 16;
  blkCnt.x = (nFace + blkSize.x - 1) / blkSize.x;
  blkCnt.y = (nSubVtx + blkSize.y - 1) / blkSize.y;
  kTessEdges<<<blkCnt, blkSize>>>(nVtx, nHe, nFace, nSub, dev_heLoops, dev_heFaces, dev_heTessIdx, dev_iuvIdxMap, dev_tessIdx);
  cudaFree(dev_heTessIdx);
  checkCUDAError("kTessEdges", __LINE__);
  
  if (useVisualize)
    cudaGraphicsUnmapResources(1, &dev_vboTessIdx, 0);
  
  if (!useVisualize)
    devAlloc(&dev_tessVtx, PPM_NVARS*(nFace*(nSub-1)*(nSub-2)/2 + nHe*(nSub-1)/2 + nVtx)*sizeof(float));
  devAlloc(&dev_tessWgt, nFace*nSubVtx*sizeof(float));

}

// allocate and initialize PPM data
void PPM::devInit() {
  devMeshInit();
  devPatchInit();
  devCoeffInit();
  devTessInit();
}

void PPM::updateCoeff() {
  dim3 blkDim(16,64), blkCnt;
  blkCnt.x = (nBasis2 + blkDim.x - 1) / blkDim.x;
  blkCnt.y = (nVtx + blkDim.y - 1) / blkDim.y;
  kUpdateCoeff<<<blkCnt,blkDim>>>(nBasis2, nVtx, bezier->dev_V, 1.0, dev_dv, dev_coeff, 0.1f);
  checkCUDAError("kUpdateCoeff", __LINE__);
}

float PPM::update() {
  if (!isBuilt)
    return 0.0f;

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
  cudaEventRecord(start, 0);

  dim3 blkDim;
  dim3 blkCnt;

  // generate/update bezier coefficients
  updateCoeff();

  // calculate new vertex positions
  if (useVisualize) {
    size_t nBytes;
    cudaGraphicsMapResources(1, &dev_vboTessVtx, 0);
    cudaGraphicsResourceGetMappedPointer((void**)&dev_tessVtx, &nBytes, dev_vboTessVtx);
  }
  /*
  if (useTessSM) {
    cudaMemset(dev_tessVtx, 0, PPM_NVARS * nFace*nSubVtx*sizeof(float));
    cudaMemset(dev_tessWgt, 0, nFace*nSubVtx*sizeof(float));

    blkDim.x = 8;
    blkDim.y = 16;
    blkDim.z = 8;
    blkCnt.x = (nHe + blkDim.x - 1) / blkDim.x;
    blkCnt.y = (nSubVtx + blkDim.y - 1) / blkDim.y;
    blkCnt.z = (PPM_NVARS + blkDim.z - 1) / blkDim.z;
    int smSize = (blkDim.z * blkDim.x * nBasis2) * sizeof(float);
    kTessVtxSM<<<blkCnt, blkDim, smSize>>>(nVtx, nHe, nSub, nSubVtx, nBasis2, degMin,
        dev_heFaces, dev_bezPatch, dev_wgtPatch, dev_coeff, dev_iuvIdxMap, dev_tessVtx, dev_tessWgt);
    checkCUDAError("kTessVtxSM", __LINE__); 

    blkDim.x = 32;
    blkDim.y = 32;
    blkDim.z = 1;
    blkCnt.x = (nFace + blkDim.x - 1) / blkDim.x;
    blkCnt.y = (nSubVtx + blkDim.y - 1) / blkDim.y;
    blkCnt.z = 1;
    kWeightScale<<<blkCnt, blkDim>>>(nFace, nSubVtx, dev_tessVtx, dev_tessWgt);
    checkCUDAError("kWeightScale", __LINE__);
  } else {
    blkDim.x = 32;
    blkDim.y = 4;
    blkDim.z = 4;
    blkCnt.x = (nFace + blkDim.x - 1) / blkDim.x;
    blkCnt.y = (nSubVtx + blkDim.y - 1) / blkDim.y;
    blkCnt.z = (PPM_NVARS + blkDim.z - 1) / blkDim.z;
    cudaMemset(dev_tessVtx, 0, PPM_NVARS * nFace*nSubVtx*sizeof(float));
    kTessVtx<<<blkCnt, blkDim>>>(nVtx, nFace, nSub, nSubVtx, nBasis2, degMin,
      dev_heFaces, dev_bezPatch, dev_wgtPatch, dev_coeff, dev_iuvIdxMap, dev_tessVtx);
    checkCUDAError("kTessVtx", __LINE__);
  }
  */
  cudaMemset(dev_tessVtx, 0, PPM_NVARS * (nFace*(nSub-1)*(nSub-2)/2 + nHe*(nSub-1)/2 + nVtx) * sizeof(float));
  
  if (nSub > 2) {
    blkDim.x = 16;
    blkDim.y = 16;
    blkDim.z = 4;
    blkCnt.x = (nFace + blkDim.x - 1) / blkDim.x;
    blkCnt.y = ((nSub-1)*(nSub-2)/2 + blkDim.y - 1) / blkDim.y;
    blkCnt.z = (PPM_NVARS + blkDim.z - 1) / blkDim.z;
    kTessVtx_Face<<<blkCnt, blkDim>>>(nVtx, nFace, nSub, nBasis2,
      dev_heFaces, dev_bezPatch, dev_wgtPatch, dev_coeff, dev_iuvInternalIdxMap, dev_tessVtx);
    checkCUDAError("kTessVtx_Face", __LINE__);
  }
  
  if (nSub > 1) {
    blkDim.x = 64;
    blkDim.y = 4;
    blkDim.z = 2;
    blkCnt.x = (nHe/2 + blkDim.x - 1) / blkDim.x;
    blkCnt.y = (nSub-1 + blkDim.y - 1) / blkDim.y;
    blkCnt.z = (PPM_NVARS + blkDim.z - 1) / blkDim.z;
    kTessVtx_Edge<<<blkCnt, blkDim>>>(nVtx, nHe, nSub, nBasis2,
      dev_heLoopsOrder,dev_bezPatch, dev_wgtPatch, dev_coeff, dev_tessVtx + PPM_NVARS*nFace*(nSub-1)*(nSub-2)/2);
    checkCUDAError("kTessVtx_Edge", __LINE__);
  }
  
  blkDim.x = 64;
  blkDim.y = 16;
  blkDim.z = 1;
  blkCnt.x = (nVtx + blkDim.x - 1) / blkDim.x;
  blkCnt.y = (PPM_NVARS + blkDim.y - 1) / blkDim.y;
  blkCnt.z = 1;
  kTessVtx_Vtx<<<blkCnt, blkDim>>>(nVtx, nSub, nBasis2,
    dev_heLoops, dev_vBndList, dev_bezPatch, dev_wgtPatch, dev_coeff, dev_tessVtx + PPM_NVARS*(nFace*(nSub-1)*(nSub-2)/2 + nHe*(nSub-1)/2));
  checkCUDAError("kTessVtx_Vtx", __LINE__);
  
  if (useVisualize)
    cudaGraphicsUnmapResources(1, &dev_vboTessVtx, 0);

  float dt;
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&dt, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  return dt;
}

// free dcel data
void PPM::devFree() {
  for (void *p : allocList)
    cudaFree(p);
  allocList.clear();

  delete bezier;
}

