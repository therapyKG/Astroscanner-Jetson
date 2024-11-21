import cuda.bindings
import cuda.bindings.driver
import cv2, cuda, timeit
import tifffile as tf

from cuda.bindings import driver, nvrtc

import numpy as np
import cupy as cp

#WORK IN PROGRESS

MEM_FLAG = driver.CUmemAttach_flags.CU_MEM_ATTACH_GLOBAL

def _cudaGetErrorEnum(error):
    if isinstance(error, driver.CUresult):
        err, name = driver.cuGetErrorName(error)
        return name if err == driver.CUresult.CUDA_SUCCESS else "<unknown>"
    elif isinstance(error, nvrtc.nvrtcResult):
        return nvrtc.nvrtcGetErrorString(error)[1]
    else:
        raise RuntimeError('Unknown error type: {}'.format(error))

def checkCudaErrors(result):
    if result[0].value:
        raise RuntimeError("CUDA error code={}({})".format(result[0].value, _cudaGetErrorEnum(result[0])))
    if len(result) == 1:
        return None
    elif len(result) == 2:
        return result[1]
    else:
        return result[1:]

def pairwise(img):
    raw = np.array(img)
    bw = np.mean(img, axis=2)
    #print(bw.shape)

def pairwise_cupy(img):
    raw = cp.array(img)
    bw = cp.mean(img, axis=2)
    #print(bw.shape)

if __name__ == "__main__":
    # read in cuda file
    kernel_code = open("stitcher.cu", 'r').read()

    img = np.array(tf.imread("../testimgs/t0.tif"))

    start = timeit.default_timer()

    # Initialize CUDA Driver API
    checkCudaErrors(driver.cuInit(0))

    # Retrieve handle for device 0
    cuDevice = checkCudaErrors(driver.cuDeviceGet(0))

    # Derive target architecture for device 0
    major = checkCudaErrors(driver.cuDeviceGetAttribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, cuDevice))
    minor = checkCudaErrors(driver.cuDeviceGetAttribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, cuDevice))
    arch_arg = bytes(f'--gpu-architecture=compute_{major}{minor}', 'ascii')

    # Create program
    prog = checkCudaErrors(nvrtc.nvrtcCreateProgram(str.encode(kernel_code), b"kernel.cu", 0, [], []))

    # Compile program
    opts = [b"--fmad=false", arch_arg]
    checkCudaErrors(nvrtc.nvrtcCompileProgram(prog, 2, opts))

    # Get PTX from compilation
    ptxSize = checkCudaErrors(nvrtc.nvrtcGetPTXSize(prog))
    ptx = b" " * ptxSize
    checkCudaErrors(nvrtc.nvrtcGetPTX(prog, ptx))

    # Create context
    context = checkCudaErrors(driver.cuCtxCreate(0, cuDevice))

    # Load PTX as module data and retrieve function
    ptx = np.char.array(ptx)
    # Note: Incompatible --gpu-architecture would be detected here
    module = checkCudaErrors(driver.cuModuleLoadData(ptx.ctypes.data))
    kernel = checkCudaErrors(driver.cuModuleGetFunction(module, b"saxpy"))

    NUM_THREADS = 512  # Threads per block
    NUM_BLOCKS = 32768  # Blocks per grid

    a = np.array([2.0], dtype=np.float32)
    #n = np.array(NUM_THREADS * NUM_BLOCKS, dtype=np.uint32)
    n = np.array(np.uint32(NUM_THREADS * NUM_BLOCKS))
    bufferSize = n * a.itemsize
    
    hX = np.random.rand(n).astype(dtype=np.float32)
    hY = np.random.rand(n).astype(dtype=np.float32)
    hOut = np.zeros(n).astype(dtype=np.float32)

    dXclass = checkCudaErrors(driver.cuMemAlloc(bufferSize))
    dYclass = checkCudaErrors(driver.cuMemAlloc(bufferSize))
    dOutclass = checkCudaErrors(driver.cuMemAlloc(bufferSize))

    stream = checkCudaErrors(driver.cuStreamCreate(0))

    checkCudaErrors(driver.cuMemcpyHtoDAsync(
    dXclass, hX.ctypes.data, bufferSize, stream
    ))
    checkCudaErrors(driver.cuMemcpyHtoDAsync(
    dYclass, hY.ctypes.data, bufferSize, stream
    ))

    # The following code example is not intuitive 
    # Subject to change in a future release
    dX = np.array([int(dXclass)], dtype=np.uint64)
    dY = np.array([int(dYclass)], dtype=np.uint64)
    dOut = np.array([int(dOutclass)], dtype=np.uint64)

    args = [a, dX, dY, dOut, n]
    args = np.array([arg.ctypes.data for arg in args], dtype = np.uint64)

    checkCudaErrors(driver.cuLaunchKernel(
    kernel,
    NUM_BLOCKS,  # grid x dim
    1,  # grid y dim
    1,  # grid z dim
    NUM_THREADS,  # block x dim
    1,  # block y dim
    1,  # block z dim
    0,  # dynamic shared memory
    stream,  # stream
    args.ctypes.data,  # kernel arguments
    0,  # extra (ignore)
    ))

    checkCudaErrors(driver.cuMemcpyDtoHAsync(
    hOut.ctypes.data, dOutclass, bufferSize, stream
    ))
    checkCudaErrors(driver.cuStreamSynchronize(stream))

    # Assert values are same after running kernel
    hZ = a * hX + hY
    if not np.allclose(hOut, hZ):
        raise ValueError("Error outside tolerance for host-device vectors")

    checkCudaErrors(driver.cuStreamDestroy(stream))
    checkCudaErrors(driver.cuMemFree(dXclass))
    checkCudaErrors(driver.cuMemFree(dYclass))
    checkCudaErrors(driver.cuMemFree(dOutclass))
    checkCudaErrors(driver.cuModuleUnload(module))
    checkCudaErrors(driver.cuCtxDestroy(context))

    print("ALL DONE")
    print("TIME = ", timeit.default_timer()-start)
    