#include "dcel.hpp"
#include "bezier.hpp"

#define FORCE_GLM_CUDA
#include "glm/glm.hpp"

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
                            const glm::vec3 *vtx, float *samp) {
  int x = blockIdx.x * blockDim.x + threadIdx.x; // grid idx
  int y = blockIdx.y * blockDim.y + threadIdx.y; // grid idx
  int vtxIdx = blockIdx.z * blockDim.z + threadIdx.z; // vert idx
  if (x >= nGrid || y >= nGrid || vtxIdx >= nVert)
    return;

  const int2 &heBnd = vHeLoopBnd[vtxIdx];
  float4 uvi = tex2DLayered<float4>(sampTexObj, x, y, heBnd.y - heBnd.x - degMin);
  int heOff = uvi.z;
  //printf("%f %f %f\n", uvi.x, uvi.y, uvi.z);

  const int4 &he0 = heLoops[heBnd.x + heOff];
  const int4 &he1 = heLoops[heBnd.x + (heOff + 1)%he0.w];
  const glm::vec3 &p0  = vtx[he0.x], &p1 = vtx[he0.y], &p2 = vtx[he1.y];
  int sIdx = x + nGrid*y + vtxIdx*nGrid*nGrid, sDim = nGrid*nGrid*nVert;
  glm::vec3 p = p0 + (p1-p0)*uvi.x + (p2-p0)*uvi.y;
  for (int k = 0; k < 3; k++)
    samp[sIdx + k*sDim] = p[k];
}

__global__ void kMeshSampleOrig(int nVert, int nGrid, int degMin,
                                cudaTextureObject_t sampTexObj,
                                const int4 *heLoops, const int2 *vHeLoopBnd,
                                const glm::vec3 *vtx, float *samp) {
  int ix = blockIdx.x * blockDim.x + threadIdx.x; // grid idx
  int iy = blockIdx.y * blockDim.y + threadIdx.y; // grid idx
  int vtxIdx = blockIdx.z * blockDim.z + threadIdx.z; // vert idx
  if (ix >= nGrid || iy >= nGrid || vtxIdx >= nVert)
    return;
  float x = 2.0f * float(ix) / nGrid - 1.0f;
  float y = 2.0f * float(iy) / nGrid - 1.0f;

  const int2 &heBnd = vHeLoopBnd[vtxIdx];
  int deg = heBnd.y - heBnd.x;
  float alpha = 2.0f*M_PI / deg;
  float th = ((y < 0) ? 2.0f*M_PI : 0.0f) + atan2f(y, x);
  float r = hypotf(x, y);
  float dTh = fmodf(th, alpha);
  int ord = floorf(th / alpha);
  float v = r*sinf(dTh) / sinf(alpha);
  float u = r*cosf(dTh) - v*cosf(alpha);
  float w = 1.0 - u - v;
  if (w < 0) {
	  float k = u + v;
	  u /= k;
	  v /= k;
	  w = 0.0;
  }

  const int4 &he0 = heLoops[heBnd.x + ord];
  const int4 &he1 = heLoops[heBnd.x + (ord+1)%he0.w];
  const glm::vec3 &p0 = vtx[he0.x], &p1 = vtx[he0.y], &p2 = vtx[he1.y];
  int sIdx = ix + nGrid*iy + vtxIdx*nGrid*nGrid, sDim = nGrid*nGrid*nVert;
  glm::vec3 p = p0 + (p1 - p0)*u + (p2 - p0)*v;
  for (int k = 0; k < 3; k++)
    samp[sIdx + k*sDim] = p[k];
}

__device__ float kPatchContrib(int degMin, int nBasis2, int nVtx, int uvIdx, int nSubVtx,
                               const int4 &he, const float2 *bezData, const float *coeff, glm::vec3 &res) {
  int dOff = he.z + (he.w*(he.w - 1) - degMin*(degMin - 1)) / 2;
  const float2 *bez = &bezData[dOff*nSubVtx*nBasis2 + uvIdx*nBasis2];
  float w = bez[0].y;
  glm::vec3 v(0.0f);
  for (int i = 0; i < nBasis2; i++) {
    for (int k = 0; k < 3; k++) {
      v[k] += bez[i].x * coeff[i + he.x*nBasis2 + k*nBasis2*nVtx];
     // printf("%f %f %d %d\n", bez[i].x, coeff[i + he.x*nBasis2 + k*nBasis2*nVtx], fIdx, uvIdx);
    }
  }
  //printf("(%f %f %f) %f (%d %d)\n", v.x, v.y, v.z, w, fIdx, uvIdx);
  res += w*v;
  return w;
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

  glm::vec3 res(0.0, 0.0, 0.0);
  float w = 0.0;
 // printf("\n");
  w += kPatchContrib(degMin, nBasis2, nVtx, UV_IDX(uv.x, uv.y), nSubVtx, he0, bezData, coeff, res);
  w += kPatchContrib(degMin, nBasis2, nVtx, UV_IDX(uv.y, nSub - uv.x - uv.y), nSubVtx, he1, bezData, coeff, res);
  w += kPatchContrib(degMin, nBasis2, nVtx, UV_IDX(nSub - uv.x - uv.y, uv.x), nSubVtx, he2, bezData, coeff, res);
  //printf("\n");

 // printf("%d:(%d,%d) -> (%f,%f,%f,%f)\n", fIdx, uv.x, uv.y, res.x, res.y, res.z, w);
 // printf("%d,%d %d,%d %d,%d\n",he0.x, he0.y, he1.x, he1.y, he2.x, he2.y);
  vtxOut[3 * (fIdx*nSubVtx + uvIdx) + 0] = res[0] / w;
  vtxOut[3 * (fIdx*nSubVtx + uvIdx) + 1] = res[1] / w;
  vtxOut[3 * (fIdx*nSubVtx + uvIdx) + 2] = res[2] / w;
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
    //printf("%d(D) %d (%d,%d,%d)\n", fSubIdx, fIdx, UV_IDX(uv.x, uv.y), UV_IDX(uv.x - 1, uv.y), UV_IDX(uv.x, uv.y - 1));
    //printf("%d(D) %d (%d,%d,%d)/%d -> (%d,%d,%d)\n", fSubIdx, fIdx, UV_IDX(uv.x,uv.y),UV_IDX(uv.x-1,uv.y),UV_IDX(uv.x,uv.y-1), nSubVtx, fIdx*nSubVtx + UV_IDX(uv.x, uv.y), fIdx*nSubVtx + UV_IDX(uv.x-1, uv.y), fIdx*nSubVtx + UV_IDX(uv.x, uv.y-1));
  }

  if (uv.x+uv.y < nSub) {
    fSubIdx = fIdx*nSubFace + UV_IDX(uv.x, uv.y);
    idxOut[3*fSubIdx + 0] = fIdx*nSubVtx + UV_IDX(uv.x, uv.y);
    idxOut[3*fSubIdx + 1] = fIdx*nSubVtx + UV_IDX(uv.x+1, uv.y);
    idxOut[3 * fSubIdx + 2] = fIdx*nSubVtx + UV_IDX(uv.x, uv.y + 1);
    //printf("%d(U) %d (%d,%d,%d)\n", fSubIdx, fIdx, UV_IDX(uv.x, uv.y), UV_IDX(uv.x + 1, uv.y), UV_IDX(uv.x, uv.y + 1));
    //printf("%d(U) %d (%d,%d,%d)/%d -> (%d,%d,%d)\n", fSubIdx, fIdx, UV_IDX(iu,iv),UV_IDX(iu+1,iv),UV_IDX(iu,iv+1), nSubVtx, fIdx*nSubVtx + UV_IDX(iu, iv), fIdx*nSubVtx + UV_IDX(iu+1, iv), fIdx*nSubVtx + UV_IDX(iu, iv+1));
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
			    printf("rescale %f %f %f\n", u, v, w);
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
void DCEL::devInit(int blkDim1d, int blkDim2d) {
  printf("uploading mesh data\n");
  cudaMalloc((void**)&dev_vList, nVtx*sizeof(glm::vec3));
  cudaMemcpy(dev_vList, &vList[0],  nVtx*sizeof(glm::vec3), cudaMemcpyHostToDevice);
  cudaMalloc((void**)&dev_heFaces, nHe*sizeof(int4));
  cudaMemcpy(dev_heFaces, &heFaces[0], nHe*sizeof(int4), cudaMemcpyHostToDevice);

  // populate the loops
  printf("sorting loops\n");
  getHeLoops();
  cudaMalloc((void**)&dev_heLoops, nHe*sizeof(int4));
  cudaMemcpy(dev_heLoops, &heLoops[0], nHe*sizeof(int4), cudaMemcpyHostToDevice);

  // fill in remaining halfedge data
  cudaMalloc((void**)&dev_vBndList, nVtx*sizeof(int2));
  cudaMemset(dev_vBndList, 0xFF, nVtx*sizeof(int2));
  dim3 blkCnt((nHe + 128 - 1) / 128);
  dim3 blkSize(128);
  kGetLoopBoundaries<<<blkCnt, blkSize>>>(nHe, dev_heLoops, dev_vBndList);
  kGetHeRootInfo<<<blkCnt, blkSize>>>(nHe, dev_vBndList, dev_heLoops, dev_heLoops);
  kGetHeRootInfo<<<blkCnt, blkSize>>>(nHe, dev_vBndList, dev_heFaces, dev_heLoops);

  // initialize the bezier patch calculator
  bezier = new Bezier<float>(8);
  cudaDeviceSynchronize();

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
  int *dev_tessIdx;
  size_t nBytes;
  cudaGraphicsMapResources(1, &dev_vboTessIdx, 0);
  cudaGraphicsResourceGetMappedPointer((void**)&dev_tessIdx, &nBytes, dev_vboTessIdx);
  blkSize.x = 64;
  blkSize.y = 16;
  blkCnt.x = (nFace + blkSize.x - 1) / blkSize.x;
  blkCnt.y = (nSubVtx + blkSize.y - 1) / blkSize.y;
  kTessEdges<<<blkCnt, blkSize>>>(nFace, nSub, dev_iuvIdxMap, dev_tessIdx);
  cudaGraphicsUnmapResources(1, &dev_vboTessIdx, 0);
  checkCUDAError("kTessEdges", __LINE__);
}

float DCEL::update() {
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
  cudaEventRecord(start, 0);

  // generate mesh sample points
  dim3 blkDim(8,8,1024/64);
  dim3 blkCnt;
  blkCnt.x = (bezier->nGrid + blkDim.x - 1)/blkDim.x;
  blkCnt.y = (bezier->nGrid + blkDim.y - 1)/blkDim.y;
  blkCnt.z = (nVtx + blkDim.z - 1)/blkDim.z;
  kMeshSample<<<blkCnt,blkDim>>>(nVtx, bezier->nGrid, degMin,
                                sampTexObj,
                                dev_heLoops, dev_vBndList,
                                dev_vList, dev_samp);
  checkCUDAError("kMeshSample", __LINE__);

  // least-squares project each dimension
  bezier->getCoeff(nVtx, dev_samp, dev_coeff);
  checkCUDAError("getCoeff", __LINE__);

  // calculate new vertex positions
  size_t nBytes;
  float *dev_tessVtx;
  cudaGraphicsMapResources(1, &dev_vboTessVtx, 0);
  cudaGraphicsResourceGetMappedPointer((void**)&dev_tessVtx, &nBytes, dev_vboTessVtx);
  blkDim.x = 128;
  blkDim.y = 8;
  blkDim.z = 1;
  blkCnt.x = (nFace + blkDim.x - 1) / blkDim.x;
  blkCnt.y = (nSubVtx + blkDim.y - 1) / blkDim.y;
  blkCnt.z = 1;
  cudaMemset(dev_tessVtx, 0, 3*nFace*nSubVtx*sizeof(float));
  kTessVtx<<<blkCnt, blkDim>>>(nVtx, nFace, nSub, nSubVtx, bezier->nBasis2, degMin,
                               dev_heFaces, dev_bezPatch, dev_coeff, dev_iuvIdxMap, dev_tessVtx);
  checkCUDAError("kTessVtx", __LINE__);
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
}
