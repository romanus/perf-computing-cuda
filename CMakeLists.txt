cmake_minimum_required (VERSION 3.8 FATAL_ERROR)

project (PerfComputingHW2 CXX CUDA)

if(NOT DEFINED COMPUTE_VERSION)
    set(COMPUTE_VERSION "compute_75")
endif()
message(STATUS "Compute version set to ${COMPUTE_VERSION}")

set(OpenCV_STATIC ON)
find_package(OpenCV REQUIRED PATHS "${CMAKE_SOURCE_DIR}/3rd_party/opencv" NO_DEFAULT_PATH)
if(OpenCV_FOUND)
    message(STATUS "OPENCV FOUND")
endif()

add_executable (CPUDemo
    src/main.cpp
    src/benchmarks.cpp
)
set_target_properties(CPUDemo PROPERTIES CXX_STANDARD 17)
target_include_directories(CPUDemo PRIVATE include)

target_link_libraries (CPUDemo
    PRIVATE
    ${OpenCV_LIBS})

add_executable (GPUDemo
    src/main.cpp
    src/benchmarks.cu
    src/filter.cu
)
set_target_properties(GPUDemo PROPERTIES CXX_STANDARD 17)
target_include_directories(GPUDemo PRIVATE include)

set(CMAKE_CONFIGURATION_TYPES "Release;" CACHE STRING "" FORCE)
set(CMAKE_CUDA_FLAGS "-Xptxas -O3 -use_fast_math -arch=${COMPUTE_VERSION} -code=${COMPUTE_VERSION}")
set_target_properties(GPUDemo PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET GPUDemo PROPERTY BUILD_RPATH ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})

target_link_libraries (GPUDemo
    PRIVATE
    ${OpenCV_LIBS}
    cudadevrt)