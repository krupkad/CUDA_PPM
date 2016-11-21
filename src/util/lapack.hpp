#ifndef UTIL_LAPACK_H
#define UTIL_LAPACK_H

#include <f2c.h>
#include <clapack.h>
#include <cublas_v2.h>

// MSVC 2013 doesnt support constexpr
#define LA_DECL(a,b) static decltype(b) a
#define LA_DEFN(a,b) decltype(b) a = b
#define LA_DECL_PTR(a,b) static decltype(b) * a
#define LA_DEFN_PTR(a,b) decltype(b) * a = b

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

#endif /* UTIL_LAPACK_H */
