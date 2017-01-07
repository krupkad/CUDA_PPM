cmake_minimum_required(VERSION 3.2)

project(robo_thesis C CXX Fortran)

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

### Set up local include and precompiled lib paths
include_directories("external/include")
if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    set(EXTERNAL_LIB_PATH "external/lib/osx")
elseif(WIN32)
    set(EXTERNAL_LIB_PATH "external/lib/win")
endif()
link_directories(${EXTERNAL_LIB_PATH})
list(APPEND CMAKE_LIBRARY_PATH "${EXTERNAL_LIB_PATH}")

### Enable C++11 for host code
set(CMAKE_CXX_STANDARD 11)

### Set up different build configurations
set(CMAKE_CONFIGURATION_TYPES Debug;RelWithDebInfo;Release
    CACHE STRING "Set configuration types" FORCE)
set(CUDA_PROPAGATE_HOST_FLAGS OFF)
set(CMAKE_CXX_FLAGS_DEBUG                 "-O0 -g")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO        "-O2 -g")
set(CMAKE_CXX_FLAGS_RELEASE               "-O3 -DNDEBUG")
list(APPEND CUDA_NVCC_FLAGS_DEBUG          -O0 -g -G)
list(APPEND CUDA_NVCC_FLAGS_RELWITHDEBINFO -O2 -g -lineinfo)
list(APPEND CUDA_NVCC_FLAGS_RELEASE        -O3 -DNDEBUG)
if (WIN32)
    set(CUDA_PROPAGATE_HOST_FLAGS ON)
    set(CMAKE_CXX_FLAGS                       "/MD /EHsc /D _CRT_SECURE_NO_WARNINGS")
    set(CMAKE_CXX_FLAGS_DEBUG                 "/Od /Zi")
    set(CMAKE_CXX_FLAGS_RELWITHDEBINFO        "/O2 /Zi")
    set(CMAKE_CXX_FLAGS_RELEASE               "/O2 /D NDEBUG")
endif()

### Detect OpenGL and GLEW
set(GLEW_INCLUDE_DIR "external/include")
set(GLEW_LIBRARY_DIR "${EXTERNAL_LIB_PATH}")
add_definitions(-DGLEW_STATIC)
find_package(GLEW)
find_package(OpenGL)

### CUDA linker options
find_package(Threads REQUIRED)
find_package(CUDA REQUIRED)
set(CUDA_ATTACH_VS_BUILD_RULE_TO_CUDA_FILE ON)
set(CUDA_SEPARABLE_COMPILATION ON)

### NanoGui options
set(NANOGUI_BUILD_EXAMPLE OFF CACHE BOOL " " FORCE)
set(NANOGUI_BUILD_PYTHON  OFF CACHE BOOL " " FORCE)
set(NANOGUI_INSTALL       OFF CACHE BOOL " " FORCE)
set(NANOGUI_BUILD_SHARED  OFF CACHE BOOL " " FORCE)
add_subdirectory(external/nanogui)
set_property(TARGET glfw glfw_objects nanogui PROPERTY FOLDER "dependencies")
add_definitions(${NANOGUI_EXTRA_DEFS})
include_directories(external/nanogui/include ${NANOGUI_EXTRA_INCS})

### LAPACK options
set(BUILD_TESTING OFF CACHE BOOL " " FORCE)
set(BUILD_COMPLEX OFF CACHE BOOL " " FORCE)
set(BUILD_COMPLEX16 OFF CACHE BOOL " " FORCE)
add_subdirectory(external/lapack)


### main project files
add_subdirectory(src)

cuda_add_executable(${CMAKE_PROJECT_NAME}
    "src/ppm.hpp"
    "src/shader.hpp"
    "src/shader.cpp"
    "src/main.cpp"
    "src/gui.hpp"
    "src/gui.cpp"
)

target_link_libraries(${CMAKE_PROJECT_NAME}
    ppm
    ${CUDA_CUBLAS_LIBRARIES}
    nanogui
    ${NANOGUI_EXTRA_LIBS}
)

add_custom_command(
    TARGET ${CMAKE_PROJECT_NAME}
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_SOURCE_DIR}/shaders
        ${CMAKE_BINARY_DIR}/shaders
)

