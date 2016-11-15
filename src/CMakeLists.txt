set(SOURCE_FILES
    "dcel_cuda.cu"
    "dcel.hpp"
    "bezier.hpp"
    )

cuda_add_library(src
    ${SOURCE_FILES}
    OPTIONS -arch=sm_50 -lineinfo
)
