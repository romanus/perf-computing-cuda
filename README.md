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
