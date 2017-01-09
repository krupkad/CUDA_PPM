#ifndef UTIL_LAPACK_H
#define UTIL_LAPACK_H

#include <cublas_v2.h>

// MSVC 2013 doesnt support constexpr
#define LA_DECL(a,b) static decltype(b) a
#define LA_DEFN(a,b) decltype(b) a = b
#define LA_DECL_PTR(a,b) static decltype(b) * a
#define LA_DEFN_PTR(a,b) decltype(b) * a = b
namespace {
extern "C" {
  typedef long int integer;
  typedef float real;
  typedef double doublereal;
  extern integer sgesdd_(char *jobz, integer *m, integer *n, real *a, 
    integer *lda, real *s, real *u, integer *ldu, real *vt, integer *ldvt, 
     real *work, integer *lwork, integer *iwork, integer *info);
  extern integer dgesdd_(char *jobz, integer *m, integer *n, doublereal *
    a, integer *lda, doublereal *s, doublereal *u, integer *ldu, 
    doublereal *vt, integer *ldvt, doublereal *work, integer *lwork, 
    integer *iwork, integer *info);
}

template <typename T>
struct LA {};

template <>
struct LA<float> {
  LA_DECL_PTR(gemm, cublasSgemm);
  LA_DECL_PTR(geam, cublasSgeam);
  LA_DECL_PTR(gesdd, sgesdd_);
  LA_DECL_PTR(dgmm, cublasSdgmm);
  LA_DECL_PTR(ger, cublasSger);
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
LA_DEFN_PTR(LA<float>::ger, cublasSger);
LA_DEFN(LA<float>::zero, 0.0f);
LA_DEFN(LA<float>::one, 1.0f);

template <>
struct LA<double> {
  LA_DECL_PTR(gemm, cublasDgemm);
  LA_DECL_PTR(geam, cublasDgeam);
  LA_DECL_PTR(gesdd, dgesdd_);
  LA_DECL_PTR(dgmm, cublasDdgmm);
  LA_DECL_PTR(ger, cublasDger);
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
LA_DEFN_PTR(LA<double>::ger, cublasDger);
LA_DEFN(LA<double>::zero, 0.0);
LA_DEFN(LA<double>::one, 1.0);

}
#endif /* UTIL_LAPACK_H */
