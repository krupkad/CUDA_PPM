set(SOURCE_FILES
    "dcel_cuda.cu"
    "dcel.cpp"
    "dcel.hpp"
    "util/lapack.hpp"
    "util/math.hpp"
    "bezier.hpp"
    )

cuda_add_library(src
    ${SOURCE_FILES}
    OPTIONS -lineinfo -std=c++11
)
