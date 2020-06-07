cmake_minimum_required (VERSION 3.8 FATAL_ERROR)

project (PerfComputingHW2 CXX CUDA)

set(OpenCV_STATIC ON)
find_package(OpenCV REQUIRED PATHS "${CMAKE_SOURCE_DIR}/3rd_party/opencv" NO_DEFAULT_PATH)
if(OpenCV_FOUND)
    message(STATUS "OPENCV FOUND")
endif()

add_executable (CPUDemo
    src/main.cpp
)
set_target_properties(CPUDemo PROPERTIES CXX_STANDARD 17)

add_executable (GPUDemo
    src/main.cu
)
set_target_properties(GPUDemo PROPERTIES CXX_STANDARD 17)

set(CMAKE_CONFIGURATION_TYPES "Release;" CACHE STRING "" FORCE)
set(CMAKE_CUDA_FLAGS "-Xptxas -O3 -use_fast_math -arch=compute_75 -code=compute_75")
set_target_properties(GPUDemo PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET GPUDemo PROPERTY BUILD_RPATH ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})

target_link_libraries (CPUDemo
    PRIVATE
    ${OpenCV_LIBS})

target_link_libraries (GPUDemo
    PRIVATE
    ${OpenCV_LIBS}
    cudadevrt)