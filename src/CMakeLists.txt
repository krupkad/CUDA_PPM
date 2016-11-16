set(SOURCE_FILES
    "dcel_cuda.cu"
    "dcel.hpp"
    "bezier.hpp"
    )

cuda_add_library(src
    ${SOURCE_FILES}
    OPTIONS -lineinfo -std=c++11
)
