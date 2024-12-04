import cupy as cp
import numpy as np
import tifffile as tf
import timeit

BINNED_K = 0
LARGE_K = 1
ALT_LARGE_K = 2
SEGMENT_K = 3

KERNEL_SELECT = 1

def use_unified_mem(bool):
    if bool:
        cp.cuda.set_allocator(cp.cuda.malloc_managed)

if __name__ == "__main__":

    # set True to use unified memory on jetson
    use_unified_mem(True)

    code = open("stitcher.cu", 'r').read()
    imgL = cp.array(tf.imread("../testimgs/0013.tif"), dtype=cp.uint16)[:, 776:1000]
    imgR = cp.array(tf.imread("../testimgs/0014.tif"), dtype=cp.uint16)[:, 0:224]

    match KERNEL_SELECT:
        # binned kernel:
        # 1 thread is 1 residual calculation, 1 thread block computes 1 residual window
        # block count equals pixel count in core (binL)
        case 0:
            start = timeit.default_timer()
            NUM_THREADS = 1024  # Threads per block
            NUM_BLOCKS = 31824 # Blocks per grid
            CLEARANCE = 16 # Margin for overlap error

            new_shape = [(int)(imgL.shape[0]/2), 2, (int)(imgL.shape[1]/2), 2]

            binL = imgL.reshape(new_shape).mean(-1).mean(1)[CLEARANCE : 500-CLEARANCE, CLEARANCE : 100-CLEARANCE].flatten()
            binR = imgR.reshape(new_shape).mean(-1).mean(1).flatten()

            kernel = cp.RawKernel(code, 'pairwise', backend='nvrtc')

            dim_row = cp.uint32(68)
            dim_col = cp.uint32(468)
            dim_window = cp.uint32(CLEARANCE * 2)

            #hX = cp.random.random_integers(65536, size=NUM_BLOCKS).astype(dtype=cp.uint16)
            #hY = cp.random.random_integers(65536, size=50000).astype(dtype=cp.uint16)
            hOut = cp.array(cp.zeros((NUM_BLOCKS * NUM_THREADS), dtype=cp.uint16))


            #time only the kernel execution
            
            kernel((NUM_BLOCKS, ), (NUM_THREADS, ), (dim_row, dim_col, dim_window, binL, binR, hOut))
            #cp.mean(hOut)
            #print(cp.max(hOut))
            print("TIME = ", timeit.default_timer()-start)

        # large kernel:
        # 1 thread is 64 residual calculations, 1 thread block computes 16 residual windows
        # TODO: RESTRUCTURE KERNEL
        case 1:

            kernel = cp.RawKernel(code, 'pairwise_line', backend='nvrtc')

            NUM_THREADS = 768
            CLEARANCE = 32
            MEM_PER_THREAD = 64
            NUM_BLOCKS = (int)(((imgL.shape[0] - CLEARANCE * 2) * (imgL.shape[1] - CLEARANCE * 2)) / ((MEM_PER_THREAD * NUM_THREADS) / (CLEARANCE * CLEARANCE * 4)))
            print(NUM_BLOCKS)

            dim_mosaic = imgL.shape
            dim_row = cp.uint32(dim_mosaic[1] - CLEARANCE * 2)
            dim_col = cp.uint32(dim_mosaic[0] - CLEARANCE * 2)
            dim_window = cp.uint32(CLEARANCE*2)

            start = timeit.default_timer()
            largeL = cp.array(imgL[CLEARANCE : 1000-CLEARANCE, CLEARANCE : 224-CLEARANCE], copy=None).flatten()
            print("TIME L = ", timeit.default_timer()-start)

            start = timeit.default_timer()
            largeR = imgR.flatten()
            print("TIME R = ", timeit.default_timer()-start)

            start = timeit.default_timer()
            temp = cp.empty((MEM_PER_THREAD * NUM_THREADS * NUM_BLOCKS), dtype=cp.uint16)
            hOut = cp.array(temp, copy=None)
            print("TIME OUT = ", timeit.default_timer()-start)

            start = timeit.default_timer()
            kernel((NUM_BLOCKS, ), (NUM_THREADS, ), (dim_row, dim_col, dim_window, largeL, largeR, hOut))

            #print(cp.min(hOut))
            print("TIME Kernel = ", timeit.default_timer()-start)

        # large kernel alternative arrangement:
        # 1 thread is 4 residual calculations, 1 thread block computes 1 residual window
        # CURRENTLY ON HOLD DUE TO NO SPEED UP OVER CASE 1
        case 2:
            start = timeit.default_timer()
            NUM_THREADS = 1024
            CLEARANCE = 32
            MEM_PER_THREAD = 4
            NUM_BLOCKS = (int)(((imgL.shape[0] - CLEARANCE * 2) * (imgL.shape[1] - CLEARANCE * 2)) / ((MEM_PER_THREAD * NUM_THREADS) / (CLEARANCE * CLEARANCE * 4)))
            print(NUM_BLOCKS)
            hOut = cp.array(cp.empty((MEM_PER_THREAD * NUM_THREADS * NUM_BLOCKS), dtype=cp.uint16))

            
            inputL = imgL[CLEARANCE : 1000-CLEARANCE, CLEARANCE : 224-CLEARANCE].flatten()
            inputR = imgR.flatten()

            kernel = cp.RawKernel(code, 'pairwise_line_alt', backend='nvrtc')
            dim_row = cp.uint32(imgL.shape[1] - CLEARANCE * 2)
            dim_col = cp.uint32(imgL.shape[0] - CLEARANCE * 2)
            dim_window = cp.uint32(CLEARANCE)

            #start = timeit.default_timer()
            
            kernel((NUM_BLOCKS, ), (NUM_THREADS, ), (dim_row, dim_col, dim_window, inputL, inputR, hOut))

            #print(cp.min(hOut))
            print("TIME = ", timeit.default_timer()-start)

        case 3:
            pass

        # CPU-only Baseline
        case 4:
            CLEARANCE = 30
            start = timeit.default_timer()
            inputL = np.array(imgL[CLEARANCE : 1000-CLEARANCE, CLEARANCE : 224-CLEARANCE].get()).flatten()
            inputR = np.array(imgR.get())
            #start = timeit.default_timer()
            residuals = [np.abs(inputR[int(i/(224-CLEARANCE*2)) : int(i/(224-CLEARANCE*2)) + CLEARANCE*2, int(i%(224-CLEARANCE*2)) : int(i%(224-CLEARANCE*2)) + CLEARANCE*2] - inputL[i]) 
                         for i in range(0,((1000-CLEARANCE*2)*(224-CLEARANCE*2)))]
            print("TIME = ", timeit.default_timer()-start)


        