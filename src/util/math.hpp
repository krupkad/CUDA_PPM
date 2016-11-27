#ifndef UTIL_MATH_H
#define UTIL_MATH_H

#ifdef __CUDA_ARCH__
#define CUDECL static inline __host__ __device__
#else
#define CUDECL static inline
#endif

CUDECL double2 operator*(double2 a, double b)
{
    return make_double2(a.x * b, a.y * b);
}
CUDECL double2 operator*(double b, double2 a)
{
    return make_double2(b * a.x, b * a.y);
}
CUDECL double2 operator*(double2 a, double2 b)
{
    return make_double2(a.x * b.x, a.y * b.y);
}
CUDECL double2 operator+(double2 a, double2 b)
{
    return make_double2(a.x + b.x, a.y + b.y);
}
CUDECL double2 operator-(double b, double2 a)
{
    return make_double2(b - a.x, b - a.y);
}
CUDECL double2 operator+(double2 a, double b)
{
  return make_double2(a.x + b, a.y + b);
}

CUDECL float2 operator*(float2 a, float b)
{
	return make_float2(a.x * b, a.y * b);
}
CUDECL float2 operator*(float b, float2 a)
{
	return make_float2(b * a.x, b * a.y);
}
CUDECL float2 operator*(float2 a, float2 b)
{
	return make_float2(a.x * b.x, a.y * b.y);
}
CUDECL float2 operator+(float2 a, float2 b)
{
	return make_float2(a.x + b.x, a.y + b.y);
}
CUDECL float2 operator+(float2 a, float b)
{
  return make_float2(a.x + b, a.y + b);
}
CUDECL float2 operator-(float b, float2 a)
{
	return make_float2(b - a.x, b - a.y);
}

#undef CUDECL
#endif /* UTIL_MATH_H */
