cuda_add_library(ppm
    "ppm_cuda.cu"
    "phys.cu"
    "ppm.cpp"
    "ppm.hpp"
    "util/lapack.hpp"
    "util/math.hpp"
    "util/error.hpp"
    "bezier.hpp"
    STATIC
    OPTIONS -lineinfo -std=c++11 -arch=sm_50
)

target_link_libraries(ppm
  lapack
  "${OPENGL_LIBRARY}"
  "${GLEW_LIBRARY}"
)

