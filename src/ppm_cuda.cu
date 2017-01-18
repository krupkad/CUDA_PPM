#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/copy.h>
#include <thrust/sort.h>

#include "ppm.hpp"
#include "bezier.hpp"
#include "util/error.hpp"

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
__global__ void kGetLoopBoundaries(int nHe, const HeData *heList, int2 *vBndList) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= nHe)
    return;

  if (i == 0 || heList[i-1].src != heList[i].src)
    vBndList[heList[i].src].x = i;
  if (i == nHe - 1 || heList[i + 1].src != heList[i].src)
    vBndList[heList[i].src].y = i+1;
}

__global__ void kMeshSample(int nVert, int nGrid, int degMin,
                            cudaTextureObject_t sampTexObj,
                            const HeData *heLoops, const int2 *vHeLoopBnd,
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

  const HeData &he0 = heLoops[heBnd.x + heOff];
  const HeData &he1 = heLoops[heBnd.x + (heOff + 1) % he0.deg];
  int sIdx = x + nGrid*y + vtxIdx*nGrid*nGrid, sDim = nGrid*nGrid*nVert;
  const float *p0, *p1, *p2;

  p0 = &vData[PPM_NVARS * he0.src];
  p1 = &vData[PPM_NVARS * he0.tgt];
  p2 = &vData[PPM_NVARS * he1.tgt];
  for (int i = 0; i < PPM_NVARS; i++)
    samp[sIdx + i * sDim] = p0[i] * w + p1[i] * u + p2[i] * v;
}

__global__ void kMeshSampleOrig(int nVert, int nGrid, int degMin,
                                cudaTextureObject_t sampTexObj,
                                const HeData *heLoops, const int2 *vHeLoopBnd,
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

  const HeData &he0 = heLoops[heBnd.x + ord];
  const HeData &he1 = heLoops[heBnd.x + (ord + 1) % he0.deg];
  int sIdx = ix + nGrid*iy + vtxIdx*nGrid*nGrid, sDim = nGrid*nGrid*nVert;
  const float *p0, *p1, *p2;

  p0 = &vtx[PPM_NVARS * he0.src];
  p1 = &vtx[PPM_NVARS * he0.tgt];
  p2 = &vtx[PPM_NVARS * he1.tgt];
  for (int i = 0; i < PPM_NVARS; i++)
    samp[sIdx + i * sDim] = p0[i] * w + p1[i] * u + p2[i] * v;
}

__global__ void kGetHeRevIdx(int nHe, int degMin, HeData *heLoops, HeData *heFaces, const int2 *vBndList) {
  int heIdx = threadIdx.x + blockIdx.x * blockDim.x;
  if (heIdx >= nHe)
    return;

  HeData &he = heLoops[heIdx];
  const int2 &bnd = vBndList[he.tgt];
  for (int i = bnd.x; i < bnd.y; i++) {
    if (heLoops[i].tgt == he.src) {
      he.revIdx = i;
      break;
    }
  }
  heFaces[he.xIdx].revIdx = heLoops[he.revIdx].xIdx;

  he.bezOff = he.ord + (he.deg*(he.deg - 1) - degMin*(degMin - 1)) / 2;
  heFaces[he.xIdx].bezOff = he.bezOff;
}

__global__ void kGetHeTessIdx(int nHe, const HeData *heLoops, int *heTessIdx) {
  int heIdx = threadIdx.x + blockIdx.x * blockDim.x;
  if (heIdx >= nHe)
    return;

  int revIdx = heLoops[heIdx].revIdx;
  if (heIdx < revIdx)
    heTessIdx[heIdx] = -heIdx-1;
  else
    heTessIdx[heIdx] = revIdx+1;
}

__device__ float patchContrib(int dIdx, int vIdx, int nBasis2, const float *bez, const float *wgt, const float *coeff, float &res) {
  bez = &bez[dIdx*nBasis2];
  float w = wgt[dIdx];
  for (int i = 0; i < nBasis2; i++)
    res += w * bez[i] * coeff[i + vIdx * nBasis2];
  return w;
}

// generate the per-face template for tessellation
#define UV_IDX(u,v) (((u)+(v)+1)*((u)+(v))/2 + (v))
__global__ void kTessVtx_Face(int nVtx, int nFace, int nSub, int nBasis2,
                        const HeData *heFaces, const float *bezData, const float *wgtData, const float *coeff,
                        const int2 *uvIdxMap, float *vDataOut) {
  int fIdx = blockIdx.x * blockDim.x + threadIdx.x;
  int uvIdx = blockIdx.y * blockDim.y + threadIdx.y;
  int dataIdx = blockIdx.z * blockDim.z + threadIdx.z;
  int nSubVtx = (nSub+1)*(nSub+2)/2;
  if (fIdx >= nFace || uvIdx >= (nSub-1)*(nSub-2)/2 || dataIdx >= PPM_NVARS)
    return;
    
  const int2 &uv = uvIdxMap[uvIdx];
  const HeData &he0 = heFaces[3 * fIdx + 0];
  const HeData &he1 = heFaces[3 * fIdx + 1];
  const HeData &he2 = heFaces[3 * fIdx + 2];

  float res = 0.0, wgt = 0.0;
  //printf("%d %d %d\n", he0.bezOff, he1.bezOff, he2.bezOff);
  wgt += patchContrib(he0.bezOff*nSubVtx + UV_IDX(uv.x, uv.y), he0.src + dataIdx*nVtx, nBasis2, bezData, wgtData, coeff, res);
  wgt += patchContrib(he1.bezOff*nSubVtx + UV_IDX(uv.y, nSub-uv.x-uv.y), he1.src + dataIdx*nVtx, nBasis2, bezData, wgtData, coeff, res);
  wgt += patchContrib(he2.bezOff*nSubVtx + UV_IDX(nSub-uv.x-uv.y, uv.x), he2.src + dataIdx*nVtx, nBasis2, bezData, wgtData, coeff, res);

  vDataOut[PPM_NVARS*(fIdx*(nSub-2)*(nSub-1)/2 + UV_IDX(uv.x-1,uv.y-1)) + dataIdx] = res / wgt;
}

__global__ void kTessVtx_Edge(int nVtx, int nHe, int nSub, int nBasis2,
                        const HeData *heFaces, const int *heTessOrder,
                        const float *bezData, const float *wgtData, const float *coeff,
                        float *vDataOut) {
  int heIdx = blockIdx.x * blockDim.x + threadIdx.x;
  int uIdx = blockIdx.y * blockDim.y + threadIdx.y;
  int dataIdx = blockIdx.z * blockDim.z + threadIdx.z;
  if (heIdx >= nHe/2 || uIdx >= nSub-1 || dataIdx >= PPM_NVARS)
    return;

  const HeData &he0 = heFaces[heTessOrder[heIdx]];
  const HeData &he1 = heFaces[he0.revIdx];
  float res = 0.0, wgt = 0.0;
  wgt += patchContrib(he0.bezOff*(nSub+1)*(nSub+2)/2 + UV_IDX(uIdx+1,0), he0.src + dataIdx*nVtx, nBasis2, bezData, wgtData, coeff, res);
  wgt += patchContrib(he1.bezOff*(nSub+1)*(nSub+2)/2 + UV_IDX(nSub-uIdx-1,0), he1.src + dataIdx*nVtx, nBasis2, bezData, wgtData, coeff, res);

  vDataOut[PPM_NVARS*(heIdx*(nSub-1) + uIdx) + dataIdx] = res / wgt;
}


__global__ void kTessVtx_Vtx(int nVtx, int nSub,  int nBasis2,
                        const HeData *heLoops, const int2 *vBndList, 
                        const float *bezData, const float *wgtData, const float *coeff,
                        float *vDataOut) {
  int vIdx = blockIdx.x * blockDim.x + threadIdx.x;
  int dataIdx = blockIdx.y * blockDim.y + threadIdx.y;
  if (vIdx >= nVtx || dataIdx >= PPM_NVARS)
    return;

  const HeData &he = heLoops[vBndList[vIdx].x];
  int dIdx = he.bezOff*(nSub+1)*(nSub+2)/2;
  
  const float *bez = &bezData[dIdx*nBasis2];
  float wgt = wgtData[dIdx], res;
  for (int i = 0; i < nBasis2; i++)
    res += wgt * bez[i] * coeff[i + (he.src + dataIdx * nVtx) * nBasis2];

  vDataOut[PPM_NVARS*vIdx + dataIdx] = res / wgt;
}

__global__ void kGetNormals(int nHe, const HeData *heLoops, float *vData) {
  int heIdx = threadIdx.x + blockIdx.x * blockDim.x;
  if (heIdx >= nHe)
    return;

  const HeData &he0 = heLoops[heIdx];
  const HeData &he1 = heLoops[heIdx - he0.ord + (he0.ord + 1) % he0.deg];
  float dx1[3], dx0[3];
  for (int i = 0; i < 3; i++) {
    dx1[i] = vData[PPM_NVARS * he1.tgt + i] - vData[PPM_NVARS * he1.src + i];
    dx0[i] = vData[PPM_NVARS * he0.tgt + i] - vData[PPM_NVARS * he0.src + i];
  }

  atomicAdd(&vData[PPM_NVARS * he0.src + 3], dx0[1] * dx1[2] - dx0[2] * dx1[1]);
  atomicAdd(&vData[PPM_NVARS * he0.src + 4], dx0[2] * dx1[0] - dx0[0] * dx1[2]);
  atomicAdd(&vData[PPM_NVARS * he0.src + 5], dx0[0] * dx1[1] - dx0[1] * dx1[0]);
}

__device__ inline int tessGetIdx(int u, int v, const HeData *heFaces, const int *heTessOrder,
                                int fIdx, int nVtx, int nHe, int nFace, int nSub) {
  const HeData &he0 = heFaces[3*fIdx+0];
  const HeData &he1 = heFaces[3*fIdx+1];
  const HeData &he2 = heFaces[3*fIdx+2];
  int heIdx1 = heTessOrder[3*fIdx+0];
  int heIdx2 = heTessOrder[3*fIdx+1];
  int heIdx3 = heTessOrder[3*fIdx+2];
  int w = nSub-u-v;
  
  if (u == 0 && v == 0)
    return nFace*(nSub-1)*(nSub-2)/2 + nHe*(nSub-1)/2 + he0.src;
  if (w == 0 && v == 0)
    return nFace*(nSub-1)*(nSub-2)/2 + nHe*(nSub-1)/2 + he1.src;
  if (u == 0 && w == 0)
    return nFace*(nSub-1)*(nSub-2)/2 + nHe*(nSub-1)/2 + he2.src;
  
  if (v == 0) {
    if (heIdx1 < nHe/2)
      return nFace*(nSub-1)*(nSub-2)/2 + heIdx1*(nSub-1) + u-1;
    else
      return nFace*(nSub-1)*(nSub-2)/2 + (nHe-heIdx1-1)*(nSub-1) + w-1;
  }
  if (w == 0) {
    if (heIdx2 < nHe/2)
      return nFace*(nSub-1)*(nSub-2)/2 + heIdx2*(nSub-1) + v-1;
    else
      return nFace*(nSub-1)*(nSub-2)/2 + (nHe-heIdx2-1)*(nSub-1) + u-1;
  }
  if (u == 0) {
    if (heIdx3 < nHe/2)
      return nFace*(nSub-1)*(nSub-2)/2 + heIdx3*(nSub-1) + w-1;
    else
      return nFace*(nSub-1)*(nSub-2)/2 + (nHe-heIdx3-1)*(nSub-1) + v-1;
  }
  
  return fIdx*(nSub-1)*(nSub-2)/2 + UV_IDX(u-1,v-1);
    
}

__global__ void kTessEdges(int nVtx, int nHe, int nFace, int nSub,
                                const HeData *heFaces, const int *heTessOrder,
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
    idxOut[3*fSubIdx + 0] = tessGetIdx(uv.x, uv.y, heFaces, heTessOrder, fIdx, nVtx, nHe, nFace, nSub);
    idxOut[3*fSubIdx + 1] = tessGetIdx(uv.x-1, uv.y, heFaces, heTessOrder, fIdx, nVtx, nHe, nFace, nSub);
    idxOut[3*fSubIdx + 2] = tessGetIdx(uv.x, uv.y-1, heFaces, heTessOrder, fIdx, nVtx, nHe, nFace, nSub);
  }

  if (uv.x+uv.y < nSub) {
    fSubIdx = fIdx*nSubFace + UV_IDX(uv.x, uv.y);
    idxOut[3*fSubIdx + 0] = tessGetIdx(uv.x, uv.y, heFaces, heTessOrder, fIdx, nVtx, nHe, nFace, nSub);
    idxOut[3*fSubIdx + 1] = tessGetIdx(uv.x+1, uv.y, heFaces, heTessOrder, fIdx, nVtx, nHe, nFace, nSub);
    idxOut[3*fSubIdx + 2] = tessGetIdx(uv.x, uv.y + 1, heFaces, heTessOrder, fIdx, nVtx, nHe, nFace, nSub);
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
  checkCUDAError("cudaCreateTextureObject", __LINE__);
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
  devAlloc(&dev_dv, 9 * nVtx*sizeof(float));
  cudaMemset(dev_dv, 0, 9*nVtx*sizeof(float));
}

void PPM::devMeshInit() {
  // populate the loops
  fprintf(stderr, "sorting loops\n");
  getHeLoops();
  
  fprintf(stderr, "uploading mesh data\n");
  devAlloc(&dev_heLoops, nHe*sizeof(HeData));
  cudaMemcpy(dev_heLoops, &heLoops[0], nHe*sizeof(HeData), cudaMemcpyHostToDevice);
  devAlloc(&dev_heFaces, nHe*sizeof(HeData));
  cudaMemcpy(dev_heFaces, &heFaces[0], nHe*sizeof(HeData), cudaMemcpyHostToDevice);
  devAlloc(&dev_vBndList, nVtx*sizeof(int2));
  cudaMemcpy(dev_vBndList, &vBndList[0], nVtx*sizeof(int2), cudaMemcpyHostToDevice);
  devAlloc(&dev_vList, PPM_NVARS*nVtx*sizeof(float));
  cudaMemcpy(dev_vList, &vList[0],  PPM_NVARS*nVtx*sizeof(float), cudaMemcpyHostToDevice);
  
  // recalculate normals
  dim3 blkDim(256);
  dim3 blkCnt((nHe + blkDim.x - 1)/blkDim.x);
  kGetHeRevIdx<<<blkCnt, blkDim>>>(nHe, degMin, dev_heLoops, dev_heFaces, dev_vBndList);
  checkCUDAError("kGetRevIdx", __LINE__);
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
  devAlloc(&dev_heTessIdx, nHe*sizeof(int));
  kGetHeTessIdx<<<blkCnt, blkSize>>>(nHe, dev_heFaces, dev_heTessIdx);
  checkCUDAError("kGetHeTessIdx", __LINE__);

  thrust::counting_iterator<int> order_itr(0);
  thrust::device_vector<int> order_vec(order_itr, order_itr+nHe);
  thrust::device_ptr<int> heTessIdx_ptr(dev_heTessIdx);
  thrust::sort_by_key(heTessIdx_ptr, heTessIdx_ptr+nHe, order_vec.begin());
  thrust::copy(order_itr, order_itr+nHe, heTessIdx_ptr);
  thrust::sort_by_key(order_vec.begin(), order_vec.end(), heTessIdx_ptr);
  
  blkSize.x = 64;
  blkSize.y = 16;
  blkCnt.x = (nFace + blkSize.x - 1) / blkSize.x;
  blkCnt.y = (nSubVtx + blkSize.y - 1) / blkSize.y;
  kTessEdges<<<blkCnt, blkSize>>>(nVtx, nHe, nFace, nSub, dev_heFaces, dev_heTessIdx, dev_iuvIdxMap, dev_tessIdx);
  checkCUDAError("kTessEdges", __LINE__);
 
  thrust::copy(order_itr, order_itr+nHe, order_vec.begin());
  thrust::sort_by_key(heTessIdx_ptr, heTessIdx_ptr+nHe, order_vec.begin());
  thrust::copy(order_vec.begin(), order_vec.end(), heTessIdx_ptr);

  if (useVisualize)
    cudaGraphicsUnmapResources(1, &dev_vboTessIdx, 0);
  
  if (!useVisualize)
    devAlloc(&dev_tessVtx, PPM_NVARS*(nFace*(nSub-1)*(nSub-2)/2 + nHe*(nSub-1)/2 + nVtx)*sizeof(float));
  devAlloc(&dev_tessWgt, nFace*nSubVtx*sizeof(float));

}

// allocate and initialize PPM data
void PPM::devInit() {
}

float PPM::update(int clickIdx, float clickForce, float dt) {
  if (!isBuilt)
    return 0.0f;

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
  cudaEventRecord(start, 0);

  dim3 blkDim;
  dim3 blkCnt;

  // generate/update bezier coefficients
  updateCoeff(clickIdx, clickForce, dt);

  // calculate new vertex positions
  if (useVisualize) {
    size_t nBytes;
    cudaGraphicsMapResources(1, &dev_vboTessVtx, 0);
    cudaGraphicsResourceGetMappedPointer((void**)&dev_tessVtx, &nBytes, dev_vboTessVtx);
  }

  if (nSub > 2) {
    blkDim.x = 16;
    blkDim.y = 4;
    blkDim.z = 2;
    blkCnt.x = (nFace + blkDim.x - 1) / blkDim.x;
    blkCnt.y = ((nSub-1)*(nSub-2)/2 + blkDim.y - 1) / blkDim.y;
    blkCnt.z = (PPM_NVARS + blkDim.z - 1) / blkDim.z;
    kTessVtx_Face<<<blkCnt, blkDim>>>(nVtx, nFace, nSub, nBasis2,
      dev_heFaces, dev_bezPatch, dev_wgtPatch, dev_coeff, dev_iuvInternalIdxMap, dev_tessVtx);
    checkCUDAError("kTessVtx_Face", __LINE__);
  }
  
  if (nSub > 1) {
    blkDim.x = 16;
    blkDim.y = 4;
    blkDim.z = 2;
    blkCnt.x = (nHe/2 + blkDim.x - 1) / blkDim.x;
    blkCnt.y = (nSub-1 + blkDim.y - 1) / blkDim.y;
    blkCnt.z = (PPM_NVARS + blkDim.z - 1) / blkDim.z;
    kTessVtx_Edge<<<blkCnt, blkDim>>>(nVtx, nHe, nSub, nBasis2,
      dev_heFaces, dev_heTessIdx, dev_bezPatch, dev_wgtPatch, dev_coeff, dev_tessVtx + PPM_NVARS*nFace*(nSub-1)*(nSub-2)/2);
    checkCUDAError("kTessVtx_Edge", __LINE__);
  }
  
  blkDim.x = 32;
  blkDim.y = 4;
  blkDim.z = 1;
  blkCnt.x = (nVtx + blkDim.x - 1) / blkDim.x;
  blkCnt.y = (PPM_NVARS + blkDim.y - 1) / blkDim.y;
  blkCnt.z = 1;
  kTessVtx_Vtx<<<blkCnt, blkDim>>>(nVtx, nSub, nBasis2,
    dev_heLoops, dev_vBndList, dev_bezPatch, dev_wgtPatch, dev_coeff, dev_tessVtx + PPM_NVARS*(nFace*(nSub-1)*(nSub-2)/2 + nHe*(nSub-1)/2));
  checkCUDAError("kTessVtx_Vtx", __LINE__);
  
  if (useVisualize)
    cudaGraphicsUnmapResources(1, &dev_vboTessVtx, 0);

  float meas_dt;
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&meas_dt, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  return meas_dt;
}

// free dcel data
void PPM::devFree() {
  for (void *p : allocList)
    cudaFree(p);
  allocList.clear();

  delete bezier;
}

