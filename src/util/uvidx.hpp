#ifndef UVIDX_UTIL
#define UVIDX_UTIL


static inline __device__ void fIuvIdxMap(int i, int2 &uv) {
  int s = floor(sqrt(2.0f*i + 0.25f) - 0.5f);
  uv.y = i - s*(s+1)/2;
  uv.x = s - uv.y;
}

static inline __device__ void fUvIdxMap(int i, float2 &uv, int nSub) {
  int2 fuv;
  fIuvIdxMap(i, fuv);
  uv.x = float(fuv.x)/nSub;
  uv.y = float(fuv.y)/nSub;
}

static inline __device__ void fIuvInternalIdxMap(int i, int2 &uv) {
  fIuvIdxMap(i, uv);
  uv.x++;
  uv.y++;
}

static inline __device__ void fUvInternalIdxMap(int i, float2 &uv, int nSub) {
  int2 fuv;
  fIuvIdxMap(i, fuv);
  uv.x = float(fuv.x+1)/nSub;
  uv.y = float(fuv.y+1)/nSub;
}


#endif /* UVIDX_UTIL */
