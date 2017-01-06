set(SOURCE_FILES
    "ppm_cuda.cu"
    "phys.cu"
    "ppm.cpp"
    "ppm.hpp"
    "util/lapack.hpp"
    "util/math.hpp"
    "bezier.hpp"
    )

cuda_add_library(src STATIC
    ${SOURCE_FILES}
    OPTIONS -lineinfo -std=c++11 -arch=sm_50
)
