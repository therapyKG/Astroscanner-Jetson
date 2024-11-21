//#include "stitcher.h"
#include <cuda_fp16.h>

extern "C" 
__global__ void saxpy(__half a, __half *x, __half *y, __half *out, size_t n){
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        out[tid] = a * x[tid] + y[tid];
    }
}

extern "C" 
__global__ void pairwise(size_t dim_row, size_t dim_col, size_t dim_window,
                         unsigned short int *x, unsigned short int *y, unsigned short int *out)
{
    unsigned int foci = (unsigned int)x[blockIdx.x];
    size_t num_row = (size_t)(blockIdx.x / dim_row);
    size_t row_idx = blockIdx.x % dim_row;
    size_t locationY = row_idx + num_row * (dim_row + dim_window);
    size_t offset = (dim_row + dim_window) * (size_t)(threadIdx.x / dim_row)
                    + (threadIdx.x % dim_row);
    unsigned int target = (unsigned int)y[locationY + offset];
    unsigned short int residual = (unsigned short int)(__vabsdiffu2(foci, target));
    out[blockIdx.x * blockDim.x + threadIdx.x] = residual;
}