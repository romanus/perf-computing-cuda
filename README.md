# HW2

## Configuration

Copy OpenCV binaries to the `3rd_party` folder and add `3rd_party/opencv/x64/vc16/bin/` to $PATH:

```
> dir /b 3rd_party\opencv
etc
include
LICENSE
OpenCVConfig-version.cmake
OpenCVConfig.cmake
samples
setup_vars_opencv4.cmd
x64
```

## Src folder

CPU code (tasks 1,2,3) is in `/src/benchmarks.cpp`.

GPU code is in `/src/benchmarks.cu` (for tasks 1,2) and `/src/filter.cu` (for task 3).

## Output

All log is in `log.txt`.

```
> mkdir build
> cd build
> cmake -G "Visual Studio 16 2019" -DCOMPUTE_VERSION=compute_75 ..
> cmake --build . --config Release
> Release\CPUDemo.exe
--------- PIXELS SUMMATION ---------
Computation time: 90[ms]
------------------------------------

--------- PIXELS REDUCTION ---------
Computation time: 182[ms]
------------------------------------

----------- CONVOLUTION ------------
Computation time: 220[ms]
------------------------------------

> Release\CPUParallelDemo.exe
--------- PIXELS SUMMATION ---------
Computation time: 121[ms]
------------------------------------

--------- PIXELS REDUCTION ---------
Computation time: 197[ms]
------------------------------------

----------- CONVOLUTION ------------
Computation time: 172[ms]
------------------------------------

PS C:\Users\trom\dev\uni\perf-computing-cuda\build> Release\CPUParallelDemo.exe
--------- PIXELS SUMMATION ---------
Computation time (1 threads): 125[ms]
Computation time (2 threads): 61[ms]
Computation time (4 threads): 41[ms]
Computation time (8 threads): 33[ms]
Computation time (16 threads): 31[ms]
Computation time (32 threads): 33[ms]
------------------------------------

--------- PIXELS REDUCTION ---------
Computation time (1 threads): 197[ms]
Computation time (2 threads): 95[ms]
Computation time (4 threads): 51[ms]
Computation time (8 threads): 42[ms]
Computation time (16 threads): 30[ms]
Computation time (32 threads): 34[ms]
------------------------------------

----------- CONVOLUTION ------------
Computation time (1 threads): 184[ms]
Computation time (2 threads): 112[ms]
Computation time (4 threads): 92[ms]
Computation time (8 threads): 88[ms]
Computation time (16 threads): 143[ms]
Computation time (32 threads): 143[ms]
Parallel algorithm matches OpenCV: true
------------------------------------

> Release\GPUDemo.exe
--------- PIXELS SUMMATION ---------
Computation time: 41[ms]
Load time: 340[ms]
Unload time: 0[ms]
------------------------------------

--------- PIXELS REDUCTION ---------
Computation time: 4[ms]
Load time: 211[ms]
Unload time: 0[ms]
------------------------------------

----------- CONVOLUTION ------------
Computation time (row filter): 9[ms]
Computation time (col filter): 4[ms]
Load time: 265[ms]
Unload time: 264[ms]
CUDA algorithm matches OpenCV: true
------------------------------------
```
