#include <cuda_fp16.h>

//sample function, mostly useless
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
    unsigned int foci = x[blockIdx.x];
    size_t num_row = (size_t)(blockIdx.x / dim_row);
    size_t row_idx = blockIdx.x % dim_row;
    size_t locationY = row_idx + num_row * (dim_row + dim_window);
    size_t offset = (dim_row + dim_window) * (size_t)(threadIdx.x / dim_row)
                    + (threadIdx.x % dim_row);
    unsigned int target = y[locationY + offset];
    unsigned short int residual = (__vabsdiffu2(foci, target));
    out[blockIdx.x * blockDim.x + threadIdx.x] = residual;
}

extern "C" 
__global__ void pairwise_line(size_t dim_row, size_t dim_col, size_t dim_window,
                         unsigned short int *x, unsigned short int *y, unsigned short int *out)
{
    unsigned int bank = threadIdx.x / dim_window;
    unsigned int foci_idx = (12 * blockIdx.x) + bank;
    unsigned int foci = x[foci_idx];

    unsigned int row_idx = ((threadIdx.x % dim_window) + (foci_idx / dim_row)) * (dim_row + dim_window);
    unsigned int row_offset = foci_idx % dim_row;
    unsigned int coalesce_idx = threadIdx.x + (foci_idx / dim_row) * (dim_row + dim_window);
    
    for (int i = 0; i < dim_window; i ++){
        //unsigned int target = y[coalesce_idx + (i * dim_window)];
        unsigned int target = y[row_idx + row_offset + i];
        unsigned short int residual = (__vabsdiffu2(foci, target));
        out[blockIdx.x * blockDim.x + threadIdx.x + i] = residual;
    }
}

extern "C" 
__global__ void pairwise_line_alt(size_t dim_row, size_t dim_col, size_t dim_window,
                         unsigned short int *x, unsigned short int *y, unsigned short int *out)
{

}