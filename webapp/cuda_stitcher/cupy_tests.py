import cupy as cp
import numpy as np
import tifffile as tf
import timeit

NUM_THREADS = 1024  # Threads per block
NUM_BLOCKS = 31824 # Blocks per grid

def use_unified_mem(bool):
    if bool:
        cp.cuda.set_allocator(cp.cuda.malloc_managed)

if __name__ == "__main__":

    # set True to use unified memory on jetson
    use_unified_mem(True)

    mempool = cp.get_default_memory_pool()

    code = open("stitcher.cu", 'r').read()
    imgL = cp.array(tf.imread("../testimgs/0013.tif"), dtype=cp.uint16).flatten()
    imgR = cp.array(tf.imread("../testimgs/0014.tif"), dtype=cp.uint16).flatten()
    print(imgL.shape)

    #print(mempool.used_bytes())
    #print(mempool.total_bytes())

    kernel = cp.RawKernel(code, 'pairwise', backend='nvrtc')

    dim_row = cp.uint32(68)
    dim_col = cp.uint32(468)
    dim_window = cp.uint32(32)

    hX = cp.random.random_integers(65536, size=NUM_BLOCKS).astype(dtype=cp.uint16)
    hY = cp.random.random_integers(65536, size=50000).astype(dtype=cp.uint16)
    hOut = cp.zeros(NUM_BLOCKS * NUM_THREADS).astype(dtype=cp.uint16)

    #time only the kernel execution
    start = timeit.default_timer()
    kernel((NUM_BLOCKS, ), (NUM_THREADS, ), (dim_row, dim_col, dim_window, hX, hY, hOut))
    #cp.mean(hOut)
    #print(cp.mean(hOut))
    print("TIME = ", timeit.default_timer()-start) 