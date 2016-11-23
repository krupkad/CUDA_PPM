#ifndef BEZIER_H
#define BEZIER_H

#include "util/lapack.hpp"
#include "util/math.hpp"
#include "util/error.hpp"

#include <cstdio>

#include <cuda_runtime.h>

#include "dcel.hpp"

template <typename T, typename T2 = typename LA<T>::T2, typename T4 = typename LA<T>::T4 >
__device__ void kBnBasis(int nBasis, const T2 &p, const T2 &np, T2 *work) {
  for (int k = 0; k < nBasis - 1; k++)
    work[k].x = work[k].y = T(0);
  work[nBasis - 1].x = work[nBasis - 1].y = T(1);

  for (int off = nBasis - 2; off >= 0; off--) {
    for (int k = off; k < nBasis - 1; k++) {
      work[k].x = work[k].x*p.x + work[k + 1].x*np.x;
      work[k].y = work[k].y*p.y + work[k + 1].y*np.y;
    }
    work[nBasis - 1].x = p.x*work[nBasis - 1].x;
    work[nBasis - 1].y = p.y*work[nBasis - 1].y;
  }
}

// make the bernstein matrix for a N*N patch of bi-degree M-1 (i.e. M functions)
// work: M*M columns, N rows to evaluate bernstein polynomials
//       via de casteljau for N evenly spaced control points
template <typename T, typename T2=typename LA<T>::T2, typename T4=typename LA<T>::T4 >
__global__ void kernCalcBFunc(int nGrid2, const T2 *gridXY, int nBasis, T2 *basis) {
  int i = blockIdx.x * blockDim.x + threadIdx.x; // grid index
  if (i >= nGrid2)
    return;

  // convert control point index to parameter
  const T2 &p = gridXY[i];
  T2 xy = T(0.5)*LA<T>::mkPair(p.x + T(1), p.y + T(1));
  T2 np = T(1) - xy;
  kBnBasis<T>(nBasis, xy, np, &basis[i*nBasis]);
}

template <typename T, typename T2=typename LA<T>::T2, typename T4=typename LA<T>::T4 >
__global__ void kernCalcBMatrix(int nGrid2, int nBasis, T2 *basis, T *A) {
  int i = blockIdx.x * blockDim.x + threadIdx.x; // basis function
  int j = blockIdx.y * blockDim.y + threadIdx.y; // basis function
  int k = blockIdx.z * blockDim.z + threadIdx.z; // control point

  if (i >= nBasis || j >= nBasis || k >= nGrid2)
    return;

  int ij = i + nBasis*j;
  A[k + ij*nGrid2] = basis[k*nBasis + i].x * basis[k*nBasis + j].y;
}

template <typename T>
struct Bezier {
  public:
    typedef typename LA<T>::T2 T2;
    typedef typename LA<T>::T4 T4;

    int nGrid, nBasis, nGrid2, nBasis2;
    cudaTextureObject_t texObj;

    T *U, *VT, *S;
    T *dev_V;
    T *dev_LLS_proj;
    T *dev_coeff;
    T2 *dev_grid;

    void free();
    void getCoeff(int nSamp, T *dev_samp, T *dev_coeff);
    void updateCoeff(int nSamp, T *dev_coeff, T* dev_dv);
    void mkGrid();

    Bezier(int nBasis, int nGrid);
    ~Bezier();

  private:

    cublasHandle_t handle;

    int svd(T *A); // svd(A) -> U,S,V
    void genSvdTex(int N);
    void mkLLSProj(); // create the projection matrix dev_LLS_proj
    T *mkBasis(int sz1, int sz2); // create the bernstein matrix

    // coefficient texture data
    cudaArray *dev_texArray;
};

template <typename T>
Bezier<T>::Bezier(int nBasis, int nGrid) : nBasis(nBasis), nGrid(nGrid) {
  cublasCreate(&handle);
  printf("created cublas handle\n"); 

  nBasis2 = nBasis*nBasis;
  nGrid2 = nGrid*nGrid;

  mkGrid();
  printf("made grid\n");

  T *A = mkBasis(1024, 32);
  printf("made basis\n");

  svd(A);
  delete A;
  printf("performed svd\n");

  mkLLSProj();
  printf("created LLS matrix\n");

  genSvdTex(10);
  printf("generate SVD textures\n");

  cudaDeviceSynchronize();
}

template  <typename T>
void Bezier<T>::mkGrid() {
  std::vector<T2> xys;
  for (int j = 0; j < nGrid; j++) {
  for (int i = 0; i < nGrid; i++) {
    T x(2 * i - nGrid + 1); x /= nGrid - 1;
    T y(2 * j - nGrid + 1); y /= nGrid - 1;

    xys.push_back(LA<T>::mkPair(x,y));
  }}
  cudaMalloc((void**)&dev_grid, nGrid2*sizeof(T2));
  cudaMemcpy(dev_grid, &xys[0], nGrid2*sizeof(T2), cudaMemcpyHostToDevice);
}

template <typename T>
Bezier<T>::~Bezier() {
  free();
}

template <typename T>
void Bezier<T>::free() {
  if (U != nullptr) {
    delete U;
    U = nullptr;
  }
  if (VT != nullptr) {
    delete VT;
    VT = nullptr;
  }
  if (S != nullptr) {
    delete S;
    S = nullptr;
  }
  if (dev_LLS_proj != nullptr) {
    cudaFree(dev_LLS_proj);
    dev_LLS_proj = nullptr;
  }
  if (dev_coeff != nullptr) {
    cudaFree(dev_coeff);
    dev_coeff = nullptr;
  }
  cublasDestroy(handle);
}

template <typename T>
void Bezier<T>::getCoeff(int nSamp, T *dev_samp, T *dev_coeff) {
  T alpha = 1.0, beta = 0.0;
  LA<T>::gemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, // column-major
              nBasis2, 3*nSamp, nGrid2,
              &alpha, dev_LLS_proj, nBasis2,
              dev_samp, nGrid2, &beta,
              dev_coeff, nBasis2);
}

template <typename T>
void Bezier<T>::updateCoeff(int nSamp, T *dev_coeff, T* dev_dv) {
  LA<T>::ger(handle, nBasis2, 3 * nSamp,
    &S[0], dev_V, 1, dev_dv, 1, dev_coeff, nBasis2);
}

template <typename T>
T *Bezier<T>::mkBasis(int sz1, int sz2) {
  T2 *dev_basis;
  cudaMalloc((void**)&dev_basis, nGrid2*nBasis*sizeof(T2));

  dim3 blkCnt1((nGrid2+sz1-1)/sz1);
  dim3 blkSz1(sz1);
  kernCalcBFunc<T><<<blkCnt1, blkSz1>>>(nGrid2,dev_grid,nBasis,dev_basis);
  checkCUDAError("kernCalcBFunc\n");

  T *dev_A;
  cudaMalloc((void**)&dev_A, nGrid2*nBasis2*sizeof(T));

  dim3 blkSz2(8,8,16);
  dim3 blkCnt2((nBasis + 8 - 1) / 8, (nBasis + 8 - 1) / 8, (nGrid2 + 16 - 1) / 16);
  kernCalcBMatrix<T><<<blkCnt2, blkSz2>>>(nGrid2, nBasis, dev_basis, dev_A);
  checkCUDAError("kernCalcBMatrix\n");

  cudaDeviceSynchronize();

  cudaFree(dev_basis);
  checkCUDAError("free basis",__LINE__);

  T *A = new T[nBasis2*nGrid2];
  cudaMemcpy(A, dev_A, nBasis2*nGrid2*sizeof(T), cudaMemcpyDeviceToHost);
  cudaFree(dev_A);

  return A;
}

template <typename T>
int Bezier<T>::svd(T *A) {
  int ret;
  T lwork_query;
  int lwork = -1;
  char job = 'A';
  LA<T>::gesdd(&job, &nGrid2, &nBasis2, NULL, &nGrid2, NULL, NULL, &nGrid2, NULL, &nBasis2,
                &lwork_query, &lwork, NULL, &ret);

  // allocate memory and perform svd
  lwork = (int)lwork_query + 1;
  U = new T[nGrid2*nGrid2];
  S = new T[nBasis2];
  VT = new T[nBasis2*nBasis2];
  T *work = new T[lwork];
  int *iwork = new int[8*nBasis2];
  LA<T>::gesdd(&job, &nGrid2, &nBasis2, A, &nGrid2, S, U, &nGrid2, VT,  &nBasis2,
                work, &lwork, iwork, &ret);
  delete work;
  delete iwork;

  // create layered texture to sample right singular vectors

  return ret;
}


// generate bernstein eigencoefficient textures
template <typename T>
void Bezier<T>::genSvdTex(int N) {
  float *texData = new T[nBasis * nBasis * N];
  for (int k = 0; k < N; k++) {
    for (int j = 0; j < nBasis; j++) {
      for (int i = 0; i < nBasis; i++) {
        texData[i + nBasis*j + nBasis2*k] = VT[j + nBasis*i + nBasis2*k];
      }
    }
  }

  cudaMalloc(&dev_V, nBasis2*nBasis2*sizeof(float));
  cudaMemcpy(dev_V, texData, nBasis2*N*sizeof(float), cudaMemcpyHostToDevice);

  printf("allocating texture memory\n");
  dev_texArray = nullptr;
  cudaChannelFormatDesc channel = cudaCreateChannelDesc<float>();
  cudaMalloc3DArray(&dev_texArray, &channel,
    make_cudaExtent(nBasis, nBasis, N), cudaArrayLayered);
  checkCUDAError("cudaMalloc3DArray", __LINE__);

  cudaMemcpy3DParms cpyParms = { 0 };
  cpyParms.srcPos = make_cudaPos(0, 0, 0);
  cpyParms.dstPos = make_cudaPos(0, 0, 0);
  cpyParms.srcPtr = make_cudaPitchedPtr(texData, nBasis*sizeof(float), nBasis, nBasis);
  cpyParms.dstArray = dev_texArray;
  cpyParms.extent = make_cudaExtent(nBasis, nBasis, N);
  cpyParms.kind = cudaMemcpyHostToDevice;
  cudaMemcpy3D(&cpyParms);
  delete texData;
  checkCUDAError("cudaMemcpy3D", __LINE__);

  printf("creating texture\n");
  cudaResourceDesc resDesc;
  memset(&resDesc, 0, sizeof resDesc);
  resDesc.resType = cudaResourceTypeArray;
  resDesc.res.array.array = dev_texArray;

  cudaTextureDesc texDesc;
  memset(&texDesc, 0, sizeof texDesc);
  texDesc.addressMode[0] = cudaAddressModeClamp;
  texDesc.addressMode[1] = cudaAddressModeClamp;
  texDesc.filterMode = cudaFilterModePoint;
  texDesc.readMode = cudaReadModeElementType;
  texDesc.normalizedCoords = 0;

  cudaCreateTextureObject(&texObj, &resDesc, &texDesc, nullptr);
  checkCUDAError("cudaCreateTextureObject", __LINE__);
}


template <typename T>
void Bezier<T>::mkLLSProj() {
  T *Sinv = new T[nBasis2];
  for (int i = 0; i < nBasis2; i++)
    Sinv[i] = 1.0/S[i];
  T *dev_VT, *dev_U, *dev_Sinv, *dev_tmp, *dev_LLS_proj_T;

  cudaMalloc((void**)&dev_VT, nBasis2*nBasis2*sizeof(T));
  cudaMemcpy(dev_VT, VT, nBasis2*nBasis2*sizeof(T), cudaMemcpyHostToDevice);

  cudaMalloc((void**)&dev_U, nGrid2*nGrid2*sizeof(T));
  cudaMemcpy(dev_U, U, nGrid2*nGrid2*sizeof(T), cudaMemcpyHostToDevice);

  cudaMalloc((void**)&dev_Sinv, nBasis2*sizeof(T));
  cudaMemcpy(dev_Sinv, Sinv, nBasis2*sizeof(T), cudaMemcpyHostToDevice);

  cudaMalloc((void**)&dev_tmp, nBasis2*nBasis2*sizeof(T));
  cudaMalloc((void**)&dev_LLS_proj_T, nBasis2*nGrid2*sizeof(T));
  cudaMalloc((void**)&dev_LLS_proj, nBasis2*nGrid2*sizeof(T));

  LA<T>::dgmm(handle, CUBLAS_SIDE_LEFT, nBasis2, nBasis2, dev_VT, nBasis2, dev_Sinv, 1,
              dev_tmp, nBasis2);
  T alpha(1.0), beta(0.0);
  LA<T>::gemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, nGrid2, nBasis2, nBasis2, &alpha, dev_U, nGrid2,
              dev_tmp, nBasis2, &beta, dev_LLS_proj_T, nGrid2);
  LA<T>::geam(handle, CUBLAS_OP_T, CUBLAS_OP_N, nBasis2, nGrid2, &alpha, dev_LLS_proj_T,
              nGrid2, &beta, dev_LLS_proj_T, nGrid2, dev_LLS_proj, nBasis2);

  delete Sinv;
  cudaFree(dev_LLS_proj_T);
  cudaFree(dev_tmp);
}

#endif /* BEZIER_H */
