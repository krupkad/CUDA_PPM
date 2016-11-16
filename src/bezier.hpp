#ifndef BEZIER_H
#define BEZIER_H

#ifdef _WIN32
#include <f2c.h>
#include <clapack.h>
#else
#include <lapacke.h>
#endif
#include <cublas_v2.h>
#include <cstdio>

#include <functional>

#include <GL/glew.h>
#include <GL/glu.h>
#include <GL/gl.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include "dcel.hpp"

inline __host__ __device__ double2 operator*(double2 a, double b)
{
    return make_double2(a.x * b, a.y * b);
}
inline __host__ __device__ double2 operator*(double b, double2 a)
{
    return make_double2(b * a.x, b * a.y);
}
inline __host__ __device__ double2 operator*(double2 a, double2 b)
{
    return make_double2(a.x * b.x, a.y * b.y);
}
inline __host__ __device__ double2 operator+(double2 a, double2 b)
{
    return make_double2(a.x + b.x, a.y + b.y);
}
inline __host__ __device__ double2 operator-(double b, double2 a)
{
    return make_double2(b - a.x, b - a.y);
}

inline __host__ __device__ float2 operator*(float2 a, float b)
{
	return make_float2(a.x * b, a.y * b);
}
inline __host__ __device__ float2 operator*(float b, float2 a)
{
	return make_float2(b * a.x, b * a.y);
}
inline __host__ __device__ float2 operator*(float2 a, float2 b)
{
	return make_float2(a.x * b.x, a.y * b.y);
}
inline __host__ __device__ float2 operator+(float2 a, float2 b)
{
	return make_float2(a.x + b.x, a.y + b.y);
}
inline __host__ __device__ float2 operator-(float b, float2 a)
{
	return make_float2(b - a.x, b - a.y);
}

// MSVC 2013 doesnt support constexpr
#ifdef _WIN32
#define LA_DECL(a,b) static decltype(b) a
#define LA_DEFN(a,b) decltype(b) a = b
#define LA_DECL_PTR(a,b) static decltype(b) * a
#define LA_DEFN_PTR(a,b) decltype(b) * a = b
#else
#define LA_DECL(a,b) static constexpr auto a = b
#define LA_DECL_PTR(a,b) LA_DECL(a,b)
#define LA_DEFN(a,b)
#define LA_DEFN_PTR(a,b)
#endif

template <typename T>
struct LA {};

template <>
struct LA<float> {
  LA_DECL_PTR(gemm, cublasSgemm);
  LA_DECL_PTR(geam, cublasSgeam);
  LA_DECL_PTR(gesdd, sgesdd_);
  LA_DECL_PTR(dgmm, cublasSdgmm);
  LA_DECL(zero, 0.0f);
  LA_DECL(one, 1.0f);
  typedef float2 T2;
  typedef float4 T4;
  static inline __host__ __device__  T2 mkPair(float x, float y) {
    return make_float2(x,y);
  }

};

LA_DEFN_PTR(LA<float>::gemm, cublasSgemm);
LA_DEFN_PTR(LA<float>::geam, cublasSgeam);
LA_DEFN_PTR(LA<float>::gesdd, sgesdd_);
LA_DEFN_PTR(LA<float>::dgmm, cublasSdgmm);
LA_DEFN(LA<float>::zero, 0.0f);
LA_DEFN(LA<float>::one, 1.0f);

template <>
struct LA<double> {
  LA_DECL_PTR(gemm, cublasDgemm);
  LA_DECL_PTR(geam, cublasDgeam);
  LA_DECL_PTR(gesdd, dgesdd_);
  LA_DECL_PTR(dgmm, cublasDdgmm);
  LA_DECL(zero, 0.0);
  LA_DECL(one, 1.0);
  typedef double2 T2;
  typedef double4 T4;
  static inline __host__ __device__  T2 mkPair(double x, double y) {
    return make_double2(x,y);
  }
};

LA_DEFN_PTR(LA<double>::gemm, cublasDgemm);
LA_DEFN_PTR(LA<double>::geam, cublasDgeam);
LA_DEFN_PTR(LA<double>::gesdd, dgesdd_);
LA_DEFN_PTR(LA<double>::dgmm, cublasDdgmm);
LA_DEFN(LA<double>::zero, 0.0);
LA_DEFN(LA<double>::one, 1.0);

// utility to fetch cuda errors and fail
void checkCUDAError(const char *msg, int line = -1) {
  cudaError_t err = cudaGetLastError();
  if (cudaSuccess != err) {
    if (line >= 0) {
      fprintf(stderr, "Line %d: ", line);
    }
    fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}

// make the bernstein matrix for a N*N patch of bi-degree M-1 (i.e. M functions)
// work: M*M columns, N rows to evaluate bernstein polynomials
//       via de casteljau for N evenly spaced control points
template <typename T, typename T2=typename LA<T>::T2, typename T4=typename LA<T>::T4 >
__global__ void kernCalcBFunc(int nGrid2, const T4 *gridUVXY, int nBasis, T2 *basis) {
  int i = blockIdx.x * blockDim.x + threadIdx.x; // grid index
  if (i >= nGrid2)
    return;

  // convert control point index to parameter
  const T4 &p = gridUVXY[i];
  T2 xy = T(0.5)*LA<T>::mkPair(p.z+T(1), p.w+T(1));
  T2 *work = &basis[i*nBasis];
  for (int k = 0; k < nBasis-1; k++)
    work[k].x = work[k].y = T(0);
  work[nBasis-1].x = work[nBasis-1].y = T(1);

  for (int off = nBasis-2; off >= 0; off--) {
    for (int k = off; k < nBasis-1; k++) {
      work[k].x = work[k].x*xy.x + work[k+1].x*(T(1) - xy.x);
      work[k].y = work[k].y*xy.y + work[k+1].y*(T(1) - xy.y);
    }
    work[nBasis-1].x = xy.x*work[nBasis-1].x;
    work[nBasis-1].y = xy.y*work[nBasis-1].y;
  }
}

template <typename T, typename T2=typename LA<T>::T2, typename T4=typename LA<T>::T4 >
__global__ void kernCalcBMatrix(int nGrid2, int nBasis, T2 *basis, T *A) {
  int ij = blockIdx.x * blockDim.x + threadIdx.x; // basis function
  int k = blockIdx.y * blockDim.y + threadIdx.y; // control point

  if (ij >= nBasis*nBasis || k >= nGrid2)
    return;

  int i = ij % nBasis, j = ij / nBasis;

  A[k + ij*nGrid2] = basis[k*nBasis + i].x * basis[k*nBasis + j].y;
}

template <typename T>
struct Bezier {
  public:
    typedef typename LA<T>::T2 T2;
    typedef typename LA<T>::T4 T4;

    int vDeg, nGrid, nBasis, nGrid2, nBasis2;

    T *U, *VT, *S;
    T *dev_LLS_proj;
    T *dev_coeff;
    T4 *dev_gridUVXY;

    void free();
    void getCoeff(int nSamp, T *dev_samp, T *dev_coeff);
    void mkGrid();

    Bezier(int vDeg);
    ~Bezier();

  private:

    cublasHandle_t handle;
    std::vector<T4> pGridXYVW;

    int svd(T *A); // svd(A) -> U,S,V
    void mkLLSProj(); // create the projection matrix dev_LLS_proj
    T *mkBasis(int sz1, int sz2); // create the bernstein matrix
};

template <typename T>
Bezier<T>::Bezier(int vDeg) : vDeg(vDeg) {
  cublasCreate(&handle);
  printf("created handle\n");


  nBasis = ((vDeg > 6) ? (vDeg + 1) : 7);
  nBasis2 = nBasis*nBasis;
  nGrid = 2*nBasis + 1;
  nGrid2 = nGrid*nGrid;

  mkGrid();
  printf("made grid\n");

  T *A = mkBasis(1024, 32);
  printf("made basis\n");

  svd(A);
  delete A;
  printf("performed svd\n");

  mkLLSProj();
  printf("created projector\n");

  cudaDeviceSynchronize();
}

template  <typename T>
void Bezier<T>::mkGrid() {
  std::vector<T4> uvxys;
  for (int i = -nBasis; i <= nBasis; i++) {
  for (int j = -nBasis; j <= nBasis; j++) {
    T x(i); x /= nBasis;
    T y(j); y /= nBasis;

    T alpha(2.0*M_PI/vDeg), th = ((j < 0) ? 2.0*M_PI : 0) + atan2(y,x), r = hypot(x,y);
    T dTh = fmod(th, alpha);

    T4 uvxy;
    // u + v*cos(alpha) = r*cos(dTh)
    // v*sin(alpha) = r*sin(dTh)
    uvxy.y = r*sin(dTh)/sin(alpha);
    uvxy.x = r*cos(dTh) - uvxy.y*cos(alpha);
    uvxy.z = x;
    uvxy.w = y;
    uvxys.push_back(uvxy);
    if (fabs(0.5 - uvxy.x - uvxy.y) > 0.5)
      printf("wout %f/%f\n", dTh, alpha);
  }}
  cudaMalloc((void**)&dev_gridUVXY, nGrid2*sizeof(T4));
  cudaMemcpy(dev_gridUVXY, &uvxys[0], nGrid2*sizeof(T4), cudaMemcpyHostToDevice);
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
T *Bezier<T>::mkBasis(int sz1, int sz2) {
  T2 *dev_basis;
  cudaMalloc((void**)&dev_basis, nGrid2*nBasis*sizeof(T2));

  dim3 blkCnt1((nGrid2+sz1-1)/sz1);
  dim3 blkSz1(sz1);
  kernCalcBFunc<T><<<blkCnt1, blkSz1>>>(nGrid2,dev_gridUVXY,nBasis,dev_basis);
  checkCUDAError("kernCalcBFunc\n");

  T *dev_A;
  cudaMalloc((void**)&dev_A, nGrid2*nBasis2*sizeof(T));

  dim3 blkSz2(sz2, sz2);
  dim3 blkCnt2((nBasis2+sz2-1)/sz2,(nGrid2+sz2-1)/sz2);
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
  printf("gesdd probe ret %d\n", ret);

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
  printf("gesdd exec ret %d\n", ret);

  return ret;
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
