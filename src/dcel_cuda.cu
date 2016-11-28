#include "dcel.hpp"
#include "bezier.hpp"

// output weights and bezier basis coefficients for tessellated subvertices of a patch
__global__ void kBezEval(int deg, int nBasis, int nSubVtx, const float2 *uvIdxMap, float2 *bzOut) {
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
    bzOut[j + nBasis*k + oIdx*nBasis*nBasis].x = sWork[j].x * sWork[k].y;
    bzOut[j + nBasis*k + oIdx*nBasis*nBasis].y = w;
  }}
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
                            const float *vtx, float *samp) {
  int x = blockIdx.x * blockDim.x + threadIdx.x; // grid idx
  int y = blockIdx.y * blockDim.y + threadIdx.y; // grid idx
  int vtxIdx = blockIdx.z * blockDim.z + threadIdx.z; // vert idx
  if (x >= nGrid || y >= nGrid || vtxIdx >= nVert)
    return;

  const int2 &heBnd = vHeLoopBnd[vtxIdx];
  float4 uvi = tex2DLayered<float4>(sampTexObj, x, y, heBnd.y - heBnd.x - degMin);
  float w = 1.0f - uvi.x - uvi.y;
  int heOff = uvi.z;

  const int4 &he0 = heLoops[heBnd.x + heOff];
  const int4 &he1 = heLoops[heBnd.x + (heOff + 1)%he0.w];
  const float *p0  = &vtx[3*he0.x], *p1 = &vtx[3*he0.y], *p2 = &vtx[3*he1.y];
  int sIdx = x + nGrid*y + vtxIdx*nGrid*nGrid, sDim = nGrid*nGrid*nVert;
  samp[sIdx + 0*sDim] = p0[0] * w + p1[0] * uvi.x + p2[0] * uvi.y;
  samp[sIdx + 1*sDim] = p0[1] * w + p1[1] * uvi.x + p2[1] * uvi.y;
  samp[sIdx + 2*sDim] = p0[2] * w + p1[2] * uvi.x + p2[2] * uvi.y;
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
  const int4 &he1 = heLoops[heBnd.x + (ord+1)%he0.w];
  const float *p0  = &vtx[3*he0.x], *p1 = &vtx[3*he0.y], *p2 = &vtx[3*he1.y];
  int sIdx = ix + nGrid*iy + vtxIdx*nGrid*nGrid, sDim = nGrid*nGrid*nVert;
  samp[sIdx + 0*sDim] = p0[0] * w + p1[0] * u + p2[0] * v;
  samp[sIdx + 1*sDim] = p0[1] * w + p1[1] * u + p2[1] * v;
  samp[sIdx + 2*sDim] = p0[2] * w + p1[2] * u + p2[2] * v;
}

__device__ void kPatchContrib(int degMin, int nBasis2, int nVtx, int uvIdx, int nSubVtx,
                               const int4 &he, const float2 *bezData, const float *coeff, float4 &res) {
  int dOff = he.z + (he.w*(he.w - 1) - degMin*(degMin - 1)) / 2;
  const float2 *bez = &bezData[dOff*nSubVtx*nBasis2 + uvIdx*nBasis2];
  float w = bez[0].y;
  res.w += w;
  for (int i = 0; i < nBasis2; i++)
    res.x += w * bez[i].x * coeff[i + he.x*nBasis2 + 0 * nBasis2*nVtx];
  for (int i = 0; i < nBasis2; i++)
    res.y += w * bez[i].x * coeff[i + he.x*nBasis2 + 1 * nBasis2*nVtx];
  for (int i = 0; i < nBasis2; i++)
    res.z += w * bez[i].x * coeff[i + he.x*nBasis2 + 2 * nBasis2*nVtx];
}

// generate the per-face template for tessellation
#define UV_IDX(u,v) (((u)+(v)+1)*((u)+(v))/2 + (v))
__global__ void kTessVtx(int nVtx, int nFace, int nSub, int nSubVtx, int nBasis2, int degMin,
                        const int4 *heFaces, const float2 *bezData, const float *coeff,
                        const int2 *uvIdxMap, float *vtxOut) {
  int fIdx = blockIdx.x * blockDim.x + threadIdx.x;
  int uvIdx = blockIdx.y * blockDim.y + threadIdx.y;
  if (fIdx >= nFace || uvIdx >= nSubVtx)
    return;

  const int4 &he0 = heFaces[3 * fIdx + 0];
  const int4 &he1 = heFaces[3 * fIdx + 1];
  const int4 &he2 = heFaces[3 * fIdx + 2];

  const int2 &uv = uvIdxMap[uvIdx];

  float4 res = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
  kPatchContrib(degMin, nBasis2, nVtx, UV_IDX(uv.x, uv.y), nSubVtx, he0, bezData, coeff, res);
  kPatchContrib(degMin, nBasis2, nVtx, UV_IDX(uv.y, nSub - uv.x - uv.y), nSubVtx, he1, bezData, coeff, res);
  kPatchContrib(degMin, nBasis2, nVtx, UV_IDX(nSub - uv.x - uv.y, uv.x), nSubVtx, he2, bezData, coeff, res);

  float *vOut = &vtxOut[3*(fIdx*nSubVtx + uvIdx)];
  vOut[0] = res.x / res.w;
  vOut[1] = res.y / res.w;
  vOut[2] = res.z / res.w;
}

__global__ void kWeightScale(int nFace, int nSubVtx, float *vtx, float *wgt) {
  int fIdx = blockIdx.x * blockDim.x + threadIdx.x;
  int uvIdx = blockIdx.y * blockDim.y + threadIdx.y;
  if (fIdx >= nFace || uvIdx >= nSubVtx)
    return;

  float w = wgt[fIdx*nSubVtx + uvIdx];

  vtx[3 * (fIdx*nSubVtx + uvIdx) + 0] /= w;
  vtx[3 * (fIdx*nSubVtx + uvIdx) + 1] /= w;
  vtx[3 * (fIdx*nSubVtx + uvIdx) + 2] /= w;
}

__global__ void kTessVtxSM(int nVtx, int nHe, int nSub, int nSubVtx, int nBasis2, int degMin,
                            const int4 *heFaces, const float2 *bezData, const float *coeff,
                            const int2 *uvIdxMap, float *vtxOut, float *wgtOut) {
  int heIdx = blockIdx.x * blockDim.x + threadIdx.x;
  int uvIdx = blockIdx.y * blockDim.y + threadIdx.y;
  if (heIdx >= nHe || uvIdx >= nSubVtx)
    return;

  const int4 &he = heFaces[heIdx];
  const int2 &uv = uvIdxMap[uvIdx];
  const int uvRot[4] = { uv.x, uv.y, nSub - uv.x - uv.y, uv.x };
  int uvIdxLoc = UV_IDX(uvRot[heIdx % 3], uvRot[heIdx % 3 + 1]);
  int dOff = he.z + (he.w*(he.w - 1) - degMin*(degMin - 1)) / 2;

  extern __shared__ float4 sTessVtxAltAll[];
  float4 *sLoc = &sTessVtxAltAll[threadIdx.x * nBasis2];
  if (threadIdx.y == 0) {
    for (int i = 0; i < nBasis2; i++)
      sLoc[i].x = coeff[i + he.x*nBasis2 + 0 * nBasis2*nVtx];
    for (int i = 0; i < nBasis2; i++)
      sLoc[i].y = coeff[i + he.x*nBasis2 + 1 * nBasis2*nVtx];
    for (int i = 0; i < nBasis2; i++)
      sLoc[i].z = coeff[i + he.x*nBasis2 + 2 * nBasis2*nVtx];
    for (int i = 0; i < nBasis2; i++)
      sLoc[i].w = 0.0f;
  }
  __syncthreads();

  float4 v = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
  v.w = bezData[dOff*nSubVtx*nBasis2 + uvIdxLoc*nBasis2].y;
  for (int i = 0; i < nBasis2; i++) {
    float b = bezData[i + dOff*nSubVtx*nBasis2 + uvIdxLoc*nBasis2].x;
    v.x += sLoc[i].x * b * v.w;
    v.y += sLoc[i].y * b * v.w;
    v.z += sLoc[i].z * b * v.w;
  }

  int fIdx = heIdx / 3;
  float *out = &vtxOut[3 * (fIdx*nSubVtx + uvIdx)];
  float *wgt = &wgtOut[fIdx*nSubVtx + uvIdx];
  atomicAdd(&out[0], v.x);
  atomicAdd(&out[1], v.y);
  atomicAdd(&out[2], v.z);
  atomicAdd(wgt, v.w);
}

__global__ void kTessVtxAltSM(int nVtx, int nHe, int nSub, int nSubVtx, int nBasis2, int degMin,
  const int4 *heFaces, const float2 *bezData, const float *coeff,
  const int2 *uvIdxMap, float *vtxOut, float *wgtOut) {
  int heIdx = blockIdx.x * blockDim.x + threadIdx.x;
  int uvIdx = blockIdx.y * blockDim.y + threadIdx.y;
  if (heIdx >= nHe)
    return;

  const int4 &he = heFaces[heIdx];
  int dOff = he.z + (he.w*(he.w - 1) - degMin*(degMin - 1)) / 2;

  extern __shared__ float4 sTessVtxAltAll2[];
  float4 *sLoc = &sTessVtxAltAll2[1 + threadIdx.x * nBasis2];
  float4 &sAcc = sTessVtxAltAll2[0];
  for (int i = threadIdx.y; i < nBasis2; i += blockDim.y) {
    sLoc[i].x = coeff[i + he.x*nBasis2 + 0 * nBasis2*nVtx];
    sLoc[i].y = coeff[i + he.x*nBasis2 + 1 * nBasis2*nVtx];
    sLoc[i].z = coeff[i + he.x*nBasis2 + 2 * nBasis2*nVtx];
  }
  __syncthreads();

  if (uvIdx >= nSubVtx)
    return;
  const int2 &uv = uvIdxMap[uvIdx];
  const int uvRot[4] = { uv.x, uv.y, nSub - uv.x - uv.y, uv.x };
  int uvIdxLoc = UV_IDX(uvRot[heIdx % 3], uvRot[heIdx % 3 + 1]);

  sAcc = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
  sAcc.w = bezData[dOff*nSubVtx*nBasis2 + uvIdxLoc*nBasis2].y;
  for (int i = 0; i < nBasis2; i++) {
    float b = bezData[i + dOff*nSubVtx*nBasis2 + uvIdxLoc*nBasis2].x;
    sAcc.x += sLoc[i].x * b * sAcc.w;
    sAcc.y += sLoc[i].y * b * sAcc.w;
    sAcc.z += sLoc[i].z * b * sAcc.w;
  }

  int fIdx = heIdx / 3;
  float *out = &vtxOut[3 * (fIdx*nSubVtx + uvIdx)];
  float *wgt = &wgtOut[fIdx*nSubVtx + uvIdx];
  atomicAdd(&out[0], sAcc.x);
  atomicAdd(&out[1], sAcc.y);
  atomicAdd(&out[2], sAcc.z);
  atomicAdd(wgt, sAcc.w);
}

__global__ void kUpdateCoeff(int nBasis2, int nSamp, float *V, float sigma, float *dv, float *coeff) {
  int bIdx = threadIdx.x + blockIdx.x * blockDim.x;
  int sIdx = threadIdx.y + blockIdx.y * blockDim.y;
  if (sIdx >= nSamp || bIdx >= nBasis2)
    return;

  float v = sigma * V[bIdx + 0 * nBasis2];
  int tIdx = bIdx + sIdx*nBasis2;
  coeff[tIdx + 0 * nSamp*nBasis2] += dv[3 * sIdx + 0] * v;
  coeff[tIdx + 1 * nSamp*nBasis2] += dv[3 * sIdx + 1] * v;
  coeff[tIdx + 2 * nSamp*nBasis2] += dv[3 * sIdx + 2] * v;
}

__global__ void kTessEdges(int nFace, int nSub, const int2 *uvIdxMap, int *idxOut) {
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
    idxOut[3*fSubIdx + 0] = fIdx*nSubVtx + UV_IDX(uv.x, uv.y);
    idxOut[3*fSubIdx + 1] = fIdx*nSubVtx + UV_IDX(uv.x-1, uv.y);
    idxOut[3*fSubIdx + 2] = fIdx*nSubVtx + UV_IDX(uv.x, uv.y-1);
  }

  if (uv.x+uv.y < nSub) {
    fSubIdx = fIdx*nSubFace + UV_IDX(uv.x, uv.y);
    idxOut[3*fSubIdx + 0] = fIdx*nSubVtx + UV_IDX(uv.x, uv.y);
    idxOut[3*fSubIdx + 1] = fIdx*nSubVtx + UV_IDX(uv.x+1, uv.y);
    idxOut[3 * fSubIdx + 2] = fIdx*nSubVtx + UV_IDX(uv.x, uv.y + 1);
  }
}

// generate sampling pattern textures
void DCEL::genSampTex() {
  printf("populating patch maps (%d-%d = %d)\n", degMin, degMax, nDeg);
  int nGrid = bezier->nGrid;
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

  printf("allocating texture memory\n");
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

  printf("creating texture\n");
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

// allocate and initialize DCEL data
void DCEL::devInit(int nBasis, int nSamp) {
  printf("uploading mesh data\n");
  cudaMalloc(&dev_vList, 3*nVtx*sizeof(float));
  cudaMemcpy(dev_vList, &vList[0],  3*nVtx*sizeof(float), cudaMemcpyHostToDevice);
  cudaMalloc(&dev_heFaces, nHe*sizeof(int4));
  cudaMemcpy(dev_heFaces, &heFaces[0], nHe*sizeof(int4), cudaMemcpyHostToDevice);

  // populate the loops
  printf("sorting loops\n");
  getHeLoops();
  cudaMalloc(&dev_heLoops, nHe*sizeof(int4));
  cudaMemcpy(dev_heLoops, &heLoops[0], nHe*sizeof(int4), cudaMemcpyHostToDevice);

  // fill in remaining halfedge data
  cudaMalloc(&dev_vBndList, nVtx*sizeof(int2));
  cudaMemset(dev_vBndList, 0xFF, nVtx*sizeof(int2));
  dim3 blkCnt((nHe + 128 - 1) / 128);
  dim3 blkSize(128);
  kGetLoopBoundaries<<<blkCnt, blkSize>>>(nHe, dev_heLoops, dev_vBndList);
  kGetHeRootInfo<<<blkCnt, blkSize>>>(nHe, dev_vBndList, dev_heLoops, dev_heLoops);
  kGetHeRootInfo<<<blkCnt, blkSize>>>(nHe, dev_vBndList, dev_heFaces, dev_heLoops);

  // initialize the bezier patch calculator
  bezier = new Bezier<float>(nBasis, nSamp);

  // build the uv index map
  printf("creating uv index map %d\n", nSubVtx);
  float2 *uvIdxMap = new float2[nSubVtx];
  int2 *iuvIdxMap = new int2[nSubVtx];
  for (int v = 0; v <= nSub; v++) {
  for (int u = 0; u <= nSub - v; u++) {
	  uvIdxMap[UV_IDX(u, v)] = make_float2(float(u) / nSub, float(v) / nSub);
	  iuvIdxMap[UV_IDX(u, v)] = make_int2(u,v);
  }}
  cudaMalloc(&dev_uvIdxMap, nSubVtx*sizeof(float2));
  cudaMemcpy(dev_uvIdxMap, uvIdxMap, nSubVtx*sizeof(float2), cudaMemcpyHostToDevice);
  cudaMalloc(&dev_iuvIdxMap, nSubVtx*sizeof(int2));
  cudaMemcpy(dev_iuvIdxMap, iuvIdxMap, nSubVtx*sizeof(int2), cudaMemcpyHostToDevice);
  delete uvIdxMap;
  delete iuvIdxMap;

  // d*(d-1)/2 - dmin*(dmin-1)/2
  printf("creating patch data\n");
  cudaMalloc(&dev_bezPatch, bezier->nBasis2*nSubVtx*((degMax + 1)*degMax / 2 - degMin*(degMin - 1) / 2)*sizeof(float2));
  blkSize.x = blkSize.y = 8;
  blkCnt.x = (nDeg+blkSize.x-1)/blkSize.x;
  blkCnt.y = (nSubVtx + blkSize.y - 1) / blkSize.y;
  int nTessSM = (bezier->nBasis + 2) * blkSize.x * blkSize.y * sizeof(float2);
  for (int d = degMin; d <= degMax; d++) {
    int dOff = d*(d - 1) - degMin*(degMin - 1);
    dOff /= 2;
    kBezEval<<<blkCnt,blkSize,nTessSM>>>(d, bezier->nBasis, nSubVtx, dev_uvIdxMap, &dev_bezPatch[dOff*nSubVtx*bezier->nBasis2]);
    checkCUDAError("kBezEval", __LINE__);
  }

  printf("creating sample data\n");
  genSampTex();
  cudaMalloc(&dev_samp, 3 * bezier->nGrid2 * nVtx * sizeof(float));
  cudaMalloc(&dev_coeff, 3 * bezier->nBasis2 * nVtx * sizeof(float));

  printf("creating edge tesselation\n");
  if (useVisualize) {
    size_t nBytes;
    cudaGraphicsMapResources(1, &dev_vboTessIdx, 0);
    cudaGraphicsResourceGetMappedPointer((void**)&dev_tessIdx, &nBytes, dev_vboTessIdx);
  } else {
    cudaMalloc(&dev_tessIdx, 3*nFace*nSubFace*sizeof(int));
  }
  blkSize.x = 64;
  blkSize.y = 16;
  blkCnt.x = (nFace + blkSize.x - 1) / blkSize.x;
  blkCnt.y = (nSubVtx + blkSize.y - 1) / blkSize.y;
  kTessEdges<<<blkCnt, blkSize>>>(nFace, nSub, dev_iuvIdxMap, dev_tessIdx);
  if (useVisualize)
    cudaGraphicsUnmapResources(1, &dev_vboTessIdx, 0);
  checkCUDAError("kTessEdges", __LINE__);


  float *dv = new float[3 * nVtx];
  for (int i = 0; i < 3 * nVtx; i++) {
    //dv[i] = 2.0f * float(rand()) / RAND_MAX - 1.0;
    //dv[i] *= .5;
    dv[i] = 0.0f;
  }
  cudaMalloc(&dev_dv, 3 * nVtx*sizeof(float));
  cudaMemcpy(dev_dv, dv, 3 * nVtx*sizeof(float), cudaMemcpyHostToDevice);
  delete dv;


  if (useSvdUpdate) {
    dim3 blkDim;
    dim3 blkCnt;

    // generate mesh sample points
    blkDim.x = 8;
    blkDim.y = 8;
    blkDim.z = 16;
    blkCnt.x = (bezier->nGrid + blkDim.x - 1) / blkDim.x;
    blkCnt.y = (bezier->nGrid + blkDim.y - 1) / blkDim.y;
    blkCnt.z = (nVtx + blkDim.z - 1) / blkDim.z;
    if (useSampTex) {
      kMeshSample<<<blkCnt, blkDim>>>(nVtx, bezier->nGrid, degMin,
        sampTexObj,
        dev_heLoops, dev_vBndList,
        dev_vList, dev_samp);
    }
    else {
      kMeshSampleOrig<<<blkCnt, blkDim>>>(nVtx, bezier->nGrid, degMin,
        sampTexObj,
        dev_heLoops, dev_vBndList,
        dev_vList, dev_samp);
    }
    checkCUDAError("kMeshSample", __LINE__);

    bezier->getCoeff(nVtx, dev_samp, dev_coeff);
    checkCUDAError("getCoeff", __LINE__);
  }

  if (!useVisualize)
    cudaMalloc(&dev_tessVtx, 3*nFace*nSubVtx*sizeof(float));
  cudaMalloc(&dev_tessWgt, nFace*nSubVtx*sizeof(float));
}

float DCEL::update() {
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
  cudaEventRecord(start, 0);

  dim3 blkDim;
  dim3 blkCnt;

  // least-squares project each dimension
  if (useSvdUpdate) {
    //cudaMemset(dev_dv, 0, 3 * nVtx*sizeof(float));
    blkDim.x = 8;
    blkDim.y = 128;
    blkDim.z = 1;
    blkCnt.x = (bezier->nBasis + blkDim.x - 1) / blkDim.x;
    blkCnt.y = (nVtx + blkDim.y - 1) / blkDim.y;
    blkCnt.z = 1;
    kUpdateCoeff<<<blkCnt, blkDim>>>(bezier->nBasis2, nVtx, bezier->dev_V, 1.0, dev_dv, dev_coeff);
    checkCUDAError("kUpdateCoeff", __LINE__);
    //bezier->updateCoeff(nVtx, dev_coeff, dev_dv);
  }
  else {
    // generate mesh sample points
    blkDim.x = 4;
    blkDim.y = 4;
    blkDim.z = 64;
    blkCnt.x = (bezier->nGrid + blkDim.x - 1) / blkDim.x;
    blkCnt.y = (bezier->nGrid + blkDim.y - 1) / blkDim.y;
    blkCnt.z = (nVtx + blkDim.z - 1) / blkDim.z;
    if (useSampTex) {
      kMeshSample<<<blkCnt, blkDim>>>(nVtx, bezier->nGrid, degMin,
        sampTexObj,
        dev_heLoops, dev_vBndList,
        dev_vList, dev_samp);
    }
    else {
      kMeshSampleOrig<<<blkCnt, blkDim>>>(nVtx, bezier->nGrid, degMin,
        sampTexObj,
        dev_heLoops, dev_vBndList,
        dev_vList, dev_samp);
    }
    checkCUDAError("kMeshSample", __LINE__);

    bezier->getCoeff(nVtx, dev_samp, dev_coeff);
    checkCUDAError("getCoeff", __LINE__);
  }

  // calculate new vertex positions
  if (useVisualize) {
    size_t nBytes;
    cudaGraphicsMapResources(1, &dev_vboTessVtx, 0);
    cudaGraphicsResourceGetMappedPointer((void**)&dev_tessVtx, &nBytes, dev_vboTessVtx);
  }
  if (useTessSM) {
    blkDim.x = 16;
    blkDim.y = 32;
    blkDim.z = 1;
    blkCnt.x = (nHe + blkDim.x - 1) / blkDim.x;
    blkCnt.y = (nSubVtx + blkDim.y - 1) / blkDim.y;
    blkCnt.z = 1;
    cudaMemset(dev_tessVtx, 0, 3 * nFace*nSubVtx*sizeof(float));
    cudaMemset(dev_tessWgt, 0, nFace*nSubVtx*sizeof(float));
    int smSize = blkDim.x * bezier->nBasis2 * sizeof(float4);
    if (useTessAltSM) {
	    kTessVtxAltSM<<<blkCnt, blkDim, smSize>>>(nVtx, nHe, nSub, nSubVtx, bezier->nBasis2, degMin,
          dev_heFaces, dev_bezPatch, dev_coeff, dev_iuvIdxMap, dev_tessVtx, dev_tessWgt);
    } else {
      kTessVtxSM<<<blkCnt, blkDim, smSize>>>(nVtx, nHe, nSub, nSubVtx, bezier->nBasis2, degMin,
          dev_heFaces, dev_bezPatch, dev_coeff, dev_iuvIdxMap, dev_tessVtx, dev_tessWgt);
    }
    checkCUDAError("kTessVtxSM", __LINE__); 

	  blkDim.x = 128;
	  blkDim.y = 8;
	  blkDim.z = 1;
	  blkCnt.x = (nFace + blkDim.x - 1) / blkDim.x;
	  blkCnt.y = (nSubVtx + blkDim.y - 1) / blkDim.y;
	  blkCnt.z = 1;
    kWeightScale<<<blkCnt, blkDim>>>(nFace, nSubVtx, dev_tessVtx, dev_tessWgt);
    checkCUDAError("kWeightScale", __LINE__);
  }
  else {
    blkDim.x = 128;
    blkDim.y = 8;
    blkDim.z = 1;
    blkCnt.x = (nFace + blkDim.x - 1) / blkDim.x;
    blkCnt.y = (nSubVtx + blkDim.y - 1) / blkDim.y;
    blkCnt.z = 1;
    cudaMemset(dev_tessVtx, 0, 3 * nFace*nSubVtx*sizeof(float));
    kTessVtx<<<blkCnt, blkDim>>>(nVtx, nFace, nSub, nSubVtx, bezier->nBasis2, degMin,
      dev_heFaces, dev_bezPatch, dev_coeff, dev_iuvIdxMap, dev_tessVtx);
    checkCUDAError("kTessVtx", __LINE__);
  }
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
void DCEL::devFree() {
  cudaFree(dev_heLoops);
  cudaFree(dev_heFaces);
  cudaFree(dev_coeff);
  cudaFree(dev_samp);
  delete bezier;
  cudaFree(dev_vList);
  cudaFree(dev_vBndList);
  cudaFree(dev_tessWgt);

  if (useVisualize) {
    cudaFree(dev_tessVtx);
    cudaFree(dev_tessIdx);
  }
}

