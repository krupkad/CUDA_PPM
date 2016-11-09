#include "dcel.hpp"
#include "bezier.hpp"

#include <unistd.h>

#define FORCE_GLM_CUDA
#include "glm/glm.hpp"

__global__ void kernBezEval(int nVtx, int nBasis, const float4 *xyIn, float *bzOut) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= nVtx)
    return;

  // each thread needs nBasis*sizeof(float2) to compute all basis functions
  // plus sizeof(float2) to store its point
  extern __shared__ float2 sAll[];
  int nSM = (nBasis + 2);
  float2 &p = sAll[threadIdx.x*nSM + 0];
  float2 &np = sAll[threadIdx.x*nSM + 1];
  float2 *sWork = &sAll[threadIdx.x*nSM + 2];

  // initialize shared memory. nothing shared between blocks so no sync
  p = make_float2(xyIn[i].x, xyIn[i].y);
  np = 1.0f - p;
  for (int k = 0; k < nBasis-1; k++)
    sWork[k].x = sWork[k].y = 0.0f;
  sWork[nBasis-1].x = sWork[nBasis-1].y = 1.0f;

  // compute all basis functions
  for (int off = nBasis-2; off >= 0; off--) {
    for (int k = off; k < nBasis-1; k++) {
      sWork[k].x = sWork[k].x*p.x + sWork[k+1].x*np.x;
      sWork[k].y = sWork[k].y*p.y + sWork[k+1].y*np.y;
    }
    sWork[nBasis-1].x = p.x*sWork[nBasis-1].x;
    sWork[nBasis-1].y = p.y*sWork[nBasis-1].y;
  }

  // tensor product and output
  for (int k = 0; k < nBasis; k++) {
  for (int j = 0; j < nBasis; j++) {
    bzOut[(j + nBasis*k) + nBasis*nBasis*i] = sWork[j].x * sWork[k].y;
  }}
}

__global__ void kernInitHeBoundaries(int N, int2 *vBndList) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N)
    return;
  vBndList[i] = make_int2(-1,-1);
}

// given a sorted list, find indices of where each block starts/ends
__global__ void kernFindHeBoundaries(int N, const int4 *heList, int2 *vBndList) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N)
    return;

  if (i == 0 || heList[i-1].x != heList[i].x)
    vBndList[heList[i].x].x = i;
  if (i == N-1 || heList[i+1].x != heList[i].x)
    vBndList[heList[i].x].y = i+1;
}

__global__ void kernSample(int nVert, int nGrid,
                            const int4 *heList, const int2 *heBndList,
                            const glm::vec3 *vtx, float *samp) {
  int2 gridIdx;
  gridIdx.x = blockIdx.x * blockDim.x + threadIdx.x; // grid idx
  gridIdx.y = blockIdx.y * blockDim.y + threadIdx.y; // grid idx
  int vtxIdx = blockIdx.z * blockDim.z + threadIdx.z; // vert idx
  if (gridIdx.x >= nGrid || gridIdx.y >= nGrid || vtxIdx >= nVert)
    return;

  float x = 2.0f * float(gridIdx.x) / nGrid - 1.0f;
  float y = 2.0f * float(gridIdx.y) / nGrid - 1.0f;

  const int2 &heBnd = heBndList[vtxIdx];
  int deg = heBnd.y - heBnd.x;
  float alpha = 2.0f*M_PI/deg;
  float th = ((y < 0) ? 2.0f*M_PI : 0.0f) + atan2f(y,x);
  float r = hypotf(x,y);
  float dTh = fmodf(th, alpha);
  int ord = floorf(th/alpha);
  float v = r*sinf(dTh)/sinf(alpha);
  float u = r*cosf(dTh) - v*cosf(alpha);
  float w = 1.0 - u - v;
  if (fabs(w-0.5f) > 0.5f) {
    u += w*u/(u+v);
    v += w*v/(u+v);
  }

  const int4 &he = heList[heBnd.x + ord];
  const glm::vec3 &p0  = vtx[he.x], &p1 = vtx[he.y], &p2 = vtx[he.z];
  int sIdx = gridIdx.x+nGrid*gridIdx.y + vtxIdx*nGrid*nGrid, sDim = nGrid*nGrid*nVert;
  glm::vec3 p = p0 + (p1-p0)*u + (p2-p0)*v;
  for (int k = 0; k < 3; k++)
    samp[sIdx + k*sDim] = p[k];
}

__global__ void kHeVtx(int nVert, int nHe, int nSubVtx, int nBasis2,
                        const int4 *heList, const float *bezIn,
                        const float *coeff, glm::vec3 *vtxOut) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= nHe*nSubVtx)
    return;
  int heIdx = idx/nSubVtx;
  int vIdx = heList[heIdx].x;

  // sbared memory size = (nBasis2 + 1)*sizeof(float)
  int smSize = (nBasis2 + 1);
  extern __shared__ float sWork[];
  float &sAccum = sWork[0 + smSize*threadIdx.x];
  float *sBez = &sWork[1 + smSize*threadIdx.x];

  // load bezier coefficients
  for (int i = 0; i < nBasis2; i++) {
    sBez[i] = bezIn[i + idx*nBasis2];
  }

  // evaluate patch
  for (int k = 0; k < 3; k++) {
    sAccum = 0.0f;
    for (int i = 0; i < nBasis2; i++) {
      sAccum += sBez[i] * coeff[i + vIdx*nBasis2 + k*nBasis2*nVert];
      //printf("%f %f\n", sBez[i], coeff[i + vIdx*nBasis2 + k*nBasis2*nVert]);
    }
    vtxOut[idx][k] = sAccum;
  }
  //printf("%d %f %f %f\n", idx, vtxOut[idx].x, vtxOut[idx].y, vtxOut[idx].z);
}


#define UV_IDX(u,v) ((u+v)*(u+v+1)/2 + v)
__global__ void kTessVtx(int nHe, int nSubVtx, const int4 *heList, const int2 *heBnd, const float4 *xyIn, const glm::vec3 *vtxIn, float *vtxOut, float *wgtOut) {
  int heIdx = blockIdx.x * blockDim.x + threadIdx.x;
  int subIdx = blockIdx.y * blockDim.y + threadIdx.y;
  if (heIdx >= nHe || subIdx >= nSubVtx)
    return;

  const float4 &xy = xyIn[heIdx*nSubVtx + subIdx];
  const int4 &he = heList[heIdx];

  const glm::vec3 &in = vtxIn[heIdx*nSubVtx + subIdx];
  int fIdx = he.w / 3;
  int oIdx = xy.w;
  float w = xy.z;
  float *out = &vtxOut[3*oIdx];
  atomicAdd(out+0, w*in[0]);
  atomicAdd(out+1, w*in[1]);
  atomicAdd(out+2, w*in[2]);
  atomicAdd(&wgtOut[oIdx], w);
  //printf("%d(%d) %d(%d): %f %f %f %f\n", fIdx, fOff, heIdx, subIdx, w, in[0],in[1],in[2]);

}

__global__ void kTessVtxWgt(int nFace, int nSubVtx, float *vtx, float *wgt) {
  int fIdx = blockIdx.x * blockDim.x + threadIdx.x;
  int subIdx = blockIdx.y * blockDim.y + threadIdx.y;
  if (fIdx >= nFace || subIdx >= nSubVtx)
    return;

  float w = wgt[fIdx*nSubVtx + subIdx];
  float *v = &vtx[3*(fIdx*nSubVtx + subIdx)];
  //printf("%d: %f %f %f %f\n", fIdx*nSubVtx + subIdx, w, v[0],v[1],v[2]);
  if (w > 0.001) {
    v[0] /= w;
    v[1] /= w;
    v[2] /= w;
  }
}

/*
__global__ void kHeTessEdges(int nHalf, const int4 *heListIn, const int4 *heListOut) {
  extern __shared__ int4 sHeList[];
  sHeList[threadIdx.x] = heList[heIdx];
  const int4 &he = sHeList[threadIdx.x];

  // get which vertex in the ring we are
  int vOrder = heIdx - heBndList[he.x].x, nSubVtx = (nSub+2)*(nSub+1)/2;
  int vDeg = heBndList[he.x].y - heBndList[he.x].x;
  int deg = vDeg;
  vDeg = (vDeg > 4) ? vDeg : 4;
  float h2 = cospif(1.0f/vDeg), h1 = 0.25f*h2;

  float vTheta = 2.0f*M_PI/deg;
  float2 csDth, csOff;
  sincosf(vTheta, &csDth.y, &csDth.x);
  sincosf(vTheta*vOrder, &csOff.y, &csOff.x);
  int fIdx = he.w / 3, fOff = he.w - 3*fIdx;


}
*/

__global__ void kHeTess(int nHalf, const int4 *heList, const int2 *heBndList,
                        int nSub, float4 *xyOut, int *idxOut) {
  int heIdx = blockIdx.x * blockDim.x + threadIdx.x;
  if (heIdx >= nHalf)
    return;

  extern __shared__ int4 sHeList[];
  sHeList[threadIdx.x] = heList[heIdx];
  const int4 &he = sHeList[threadIdx.x];

  // get which vertex in the ring we are
  int vOrder = heIdx - heBndList[he.x].x, nSubVtx = (nSub+2)*(nSub+1)/2;
  int vDeg = heBndList[he.x].y - heBndList[he.x].x;
  int deg = vDeg;
  vDeg = (vDeg > 4) ? vDeg : 4;
  float h2 = cospif(1.0f/vDeg), h1 = 0.25f*h2;

  float vTheta = 2.0f*M_PI/deg;
  float2 csDth, csOff;
  sincosf(vTheta, &csDth.y, &csDth.x);
  sincosf(vTheta*vOrder, &csOff.y, &csOff.x);
  int fIdx = he.w / 3, fOff = he.w - 3*fIdx;

  int uvw[4];
  int fSubIdx;
  for (int iu = 0; iu < nSub+1; iu++) {
  for (int iv = 0; iv < nSub-iu+1; iv++) {

    float u = float(iu)/nSub;
    float v = float(iv)/nSub;

    uvw[0] = iu;
    uvw[1] = iv;
    uvw[2] = nSub-iu-iv;
    uvw[3] = iu;

    float x1 = u + v*csDth.x, y1 = v*csDth.y;
    float x = x1*csOff.x - y1*csOff.y, y = x1*csOff.y + y1*csOff.x;
    float r = hypotf(x,y);
    float h = (r - h1) / (h2 - h1);
    float s = rsqrtf(1.0f - h) - rsqrtf(h);
    float w = (r < h1) ? 1.0f : ((r > h2) ? 0.0f : 1.0f/(1.0f + expf(2.0f*s)));

    xyOut[heIdx*nSubVtx + UV_IDX(iu, iv)] =
        make_float4(.5f*x + .5f, .5f*y + .5f, w, fIdx*nSubVtx + UV_IDX(uvw[fOff], uvw[fOff+1]));
    //printf("%d(%d) %d -> (%f %f) %d\n", fIdx, fOff, UV_IDX(iu,iv), r, w,fIdx*nSubVtx + UV_IDX(uvw[fOff], uvw[fOff+1]));

    if (iu > 0 && iv > 0 && fOff == 0) {
      fSubIdx = fIdx*nSub*nSub + UV_IDX(iu-1, iv-1) + nSubVtx - nSub - 1;
      idxOut[3*fSubIdx + 0] = fIdx*nSubVtx + UV_IDX(iu, iv);
      idxOut[3*fSubIdx + 1] = fIdx*nSubVtx + UV_IDX(iu-1, iv);
      idxOut[3*fSubIdx + 2] = fIdx*nSubVtx + UV_IDX(iu, iv-1);
      //printf("%d(D) %d (%d,%d,%d)/%d -> (%d,%d,%d)\n", fSubIdx, fIdx, UV_IDX(iu,iv),UV_IDX(iu-1,iv),UV_IDX(iu,iv-1), nSubVtx, fIdx*nSubVtx + UV_IDX(iu, iv), fIdx*nSubVtx + UV_IDX(iu-1, iv), fIdx*nSubVtx + UV_IDX(iu, iv-1));
    }

    if (iu+iv < nSub && fOff == 0) {
      fSubIdx = fIdx*nSub*nSub + UV_IDX(iu, iv);
      idxOut[3*fSubIdx + 0] = fIdx*nSubVtx + UV_IDX(iu, iv);
      idxOut[3*fSubIdx + 1] = fIdx*nSubVtx + UV_IDX(iu+1, iv);
      idxOut[3*fSubIdx + 2] = fIdx*nSubVtx + UV_IDX(iu, iv+1);
      //printf("%d(U) %d (%d,%d,%d)/%d -> (%d,%d,%d)\n", fSubIdx, fIdx, UV_IDX(iu,iv),UV_IDX(iu+1,iv),UV_IDX(iu,iv+1), nSubVtx, fIdx*nSubVtx + UV_IDX(iu, iv), fIdx*nSubVtx + UV_IDX(iu+1, iv), fIdx*nSubVtx + UV_IDX(iu, iv+1));
    }
  }}
}

// allocate and initialize DCEL data
void DCEL::devInit(int blkDim1d, int blkDim2d) {
  // get the vertex count
  nVtx = vList.size();
  nHe = heList.size();
  nFace = triList.size();

  dim3 blkCnt1d((nVtx + blkDim1d - 1) / blkDim1d);
  dim3 blkSize1d(blkDim1d);

  printf("uploading mesh data\n");
  cudaMalloc((void**)&dev_vList, nVtx*sizeof(glm::vec3));
  cudaMemcpy(dev_vList, &vList[0],  nVtx*sizeof(glm::vec3), cudaMemcpyHostToDevice);
  cudaMalloc((void**)&dev_vBndList, nVtx*sizeof(int2));
  cudaMemcpy(dev_vBndList, &vBndList[0], nVtx*sizeof(int2), cudaMemcpyHostToDevice);
  cudaMalloc((void**)&dev_heList, nHe*sizeof(int4));
  cudaMemcpy(dev_heList, &heList[0],  nHe*sizeof(int4), cudaMemcpyHostToDevice);
  cudaMalloc((void**)&dev_triList, nFace*sizeof(glm::ivec3));
  cudaMemcpy(dev_triList, &triList[0],  nFace*sizeof(glm::ivec3), cudaMemcpyHostToDevice);

  // find halfedge root spans
  blkCnt1d.x = (nHe + blkDim1d - 1) / blkDim1d;
  kernFindHeBoundaries<<<blkCnt1d, blkSize1d>>>(nHe, dev_heList, dev_vBndList);

  // initialize the bezier patch calculator
  bezier = new Bezier<float>(6);
  cudaDeviceSynchronize();

  // tesselation controls
  nSub = 1;
  nSubFace = nSub*nSub;
  nSubVtx = (nSub+1)*(nSub+2)/2;

  // tessellation - get patch coordinates
  cudaMalloc((void**)&dev_tessXY, nHe*nSubVtx*sizeof(float4));
  int *dev_tessIdx;
  cudaMalloc((void**)&dev_tessIdx, 3*nFace*nSubVtx*sizeof(int));
  int nTessSM = sizeof(int4)*blkDim1d;
  kHeTess<<<blkCnt1d,blkDim1d,nTessSM>>>(nHe, dev_heList, dev_vBndList, nSub, dev_tessXY, dev_tessIdx);
  tessIdx = new int[3*nSubFace*nFace];
  cudaMemcpy(tessIdx, dev_tessIdx, 3*nFace*nSubFace*sizeof(int), cudaMemcpyDeviceToHost);
  checkCUDAError("tessGen", __LINE__);

  // tesselation - evaluate bernstein basis
  nTessSM = (bezier->nBasis + 2) * blkDim1d * sizeof(float2);
  blkCnt1d.x = (nSubVtx*nHe + blkDim1d - 1) / blkDim1d;
  cudaMalloc((void**)&dev_tessBez, bezier->nBasis2*nHe*nSubVtx*sizeof(float));
  cudaMemset(dev_tessBez, 0, bezier->nBasis2*nHe*nSubVtx*sizeof(float));
  kernBezEval<<<blkCnt1d,blkDim1d,nTessSM>>>(nSubVtx*nHe, bezier->nBasis, dev_tessXY, dev_tessBez);
  checkCUDAError("tessBezEval", __LINE__);
  cudaDeviceSynchronize();

  // tesselation - reduce to vertex contributions
  cudaMalloc((void**)&dev_samp, 3*nVtx*bezier->nGrid2*sizeof(float));
  cudaMalloc((void**)&dev_coeff, 3*nVtx*bezier->nBasis2*sizeof(float));
  cudaMalloc((void**)&dev_tessAllVtx, nHe*nSubVtx*sizeof(glm::vec3));
  cudaMalloc((void**)&dev_tessVtx, 3*nFace*nSubVtx*sizeof(float));
  cudaMalloc((void**)&dev_tessWgt, nFace*nSubVtx*sizeof(float));
  tessVtx = new float[3*nFace*nSubVtx];
}

void DCEL::sample() {
  //dim3 blkDim(16,64);
  dim3 blkDim(8,8,1024/64);
  dim3 blkCnt;
  blkCnt.x = (bezier->nGrid + blkDim.x - 1)/blkDim.x;
  blkCnt.y = (bezier->nGrid + blkDim.y - 1)/blkDim.y;
  blkCnt.z = (nVtx + blkDim.z - 1)/blkDim.z;

  // generate points on the mesh
  kernSample<<<blkCnt,blkDim>>>(nVtx, bezier->nGrid,
                                dev_heList, dev_vBndList,
                                dev_vList, dev_samp);
  checkCUDAError("sample", __LINE__);
  /*float *sf = new float[3*bezier->nGrid2 * nVtx];
  cudaMemcpy(sf, dev_samp, bezier->nGrid2 * nVtx * sizeof(float), cudaMemcpyDeviceToHost);
  for (int i = 0; i < nVtx * bezier->nGrid2; i++)
    printf("%f ", sf[i]);
  printf("\n");*/

  // least-squares project each dimension
  bezier->getCoeff(vList.size(), dev_samp, dev_coeff);

  blkDim.x = 32;
  blkDim.y = 1;
  blkDim.z = 1;
  blkCnt.x = (nHe*nSubVtx + blkDim.x - 1) / blkDim.x;
  blkCnt.y = 1;
  blkCnt.z = 1;
  int nTessSM = (bezier->nBasis2 + 1) * blkDim.x * sizeof(float);
  kHeVtx<<<blkCnt,blkDim,nTessSM>>>(nVtx, nHe, nSubVtx, bezier->nBasis2,
                                              dev_heList, dev_tessBez, dev_coeff, dev_tessAllVtx);
  checkCUDAError("kHeVtx", __LINE__);

  blkDim.x = 128;
  blkDim.y = 8;
  blkCnt.x = (nHe + blkDim.x - 1) / blkDim.x;
  blkCnt.y = (nSubVtx + blkDim.y - 1) / blkDim.y;
  cudaMemset(dev_tessVtx, 0, 3*nFace*nSubVtx*sizeof(float));
  cudaMemset(dev_tessWgt, 0, nFace*nSubVtx*sizeof(float));
  kTessVtx<<<blkCnt,blkDim>>>(nHe, nSubVtx, dev_heList, dev_vBndList, dev_tessXY, dev_tessAllVtx, dev_tessVtx, dev_tessWgt);
  checkCUDAError("kTessVtx", __LINE__);

  blkCnt.x = (nFace + blkDim.x - 1) / blkDim.x;
  kTessVtxWgt<<<blkCnt,blkDim>>>(nFace, nSubVtx, dev_tessVtx, dev_tessWgt);
  cudaMemcpy(tessVtx, dev_tessVtx, 3*nFace*nSubVtx*sizeof(float), cudaMemcpyDeviceToHost);
  checkCUDAError("kTessVtxWgt", __LINE__);

  //glBindBuffer(GL_ARRAY_BUFFER, vboVtxList);
  //glBufferData(GL_ARRAY_BUFFER, 3*vCnt*sizeof(float), vboVtxListBuf, GL_STATIC_DRAW);
}

// free dcel data
void DCEL::devFree() {
  cudaFree(dev_heList);
  cudaFree(dev_coeff);
  cudaFree(dev_samp);
  delete bezier;
  cudaFree(dev_vList);
  cudaFree(dev_vBndList);
  cudaFree(dev_uvGrid);
  cudaFree(dev_tessBez);
  cudaFree(dev_tessAllVtx);
  cudaFree(dev_tessXY);
  delete tessVtx;
  delete tessIdx;
}

void DCEL::visUpdate() {
  //size_t num_bytes;
  //float4 *dev_vtx;
  //cudaGraphicsMapResources(1, &cuVtxResource, 0);
  //cudaGraphicsResourceGetMappedPointer((void**)&dev_vtx,
  //                                       &num_bytes,
  //                                       cuVtxResource);

  /*
  int heCnt = heList.size();
  int blkSz = 1024;
  dim3 blkCnt((heCnt + blkSz - 1)/blkSz);
  kernFindHeTriangles<<<blkCnt, blkSize>>>(heCnt, dev_vList, dev_heList, dev_heVtxList);
  cudaMemcpy(&triList[0], dev_triList,  F*sizeof(glm::ivec3), cudaMemcpyDeviceToHost);
  */

  sample();
  cudaDeviceSynchronize();

  /*
  cudaMemcpy(&vList[0], vList, vCount*sizeof(glm::vec3), cudaMemcpyDeviceToHost);
  for(int i = 0; i < vList.size(); i++)
    memcpy(&vboData[6*i], &vList[i], 3*sizeof(float));
  checkCUDAError("download", __LINE__);

  glBindBuffer(GL_ARRAY_BUFFER, vboVtx);
  glBufferData(GL_ARRAY_BUFFER, 3*vList.size()*sizeof(float), vboVtxData, GL_STATIC_DRAW);

  cudaThreadSynchronize();
  checkCUDAError("sync", __LINE__);
  */

  //cudaGraphicsUnmapResources(1, &cuVtxResource, 0);
}
