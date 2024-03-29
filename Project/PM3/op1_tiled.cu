#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"

#define TILE_WIDTH 16

__global__ void conv_forward_kernel(float *y, const float *x, const float *k, const int B, const int M, const int C, const int H, const int W, const int K)
{
    /*
    Modify this function to implement the forward pass described in Chapter 16.
    We have added an additional dimension to the tensors to support an entire mini-batch
    The goal here is to be correct AND fast.

    Function paramter definitions:
    y - output
    x - input
    k - kernel
    B - batch_size (number of images in x)
    M - number of output feature maps
    C - number of input feature maps
    H - input height dimension
    W - input width dimension
    K - kernel height and width (K x K)
    */

    const int H_out = H - K + 1;
    const int W_out = W - K + 1;
    const int W_grid = (W_out + TILE_WIDTH - 1) / TILE_WIDTH;
    const int BLOCK_WIDTH = TILE_WIDTH + K - 1;
    extern __shared__ float shared_X[];

    // (void)H_out; // silence declared but never referenced warning. remove this line when you start working
    // (void)W_out; // silence declared but never referenced warning. remove this line when you start working

    // We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    // An example use of these macros:
    // float a = y4d(0,0,0,0)
    // y4d(0,0,0,0) = a

#define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
#define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
#define k4d(i3, i2, i1, i0) k[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    // Insert your GPU convolution kernel code here
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tIdx = tx + ty * BLOCK_WIDTH;
    int m = blockIdx.x;
    int b = blockIdx.z;
    int h = (blockIdx.y / W_grid) * TILE_WIDTH + ty;
    int w = (blockIdx.y % W_grid) * TILE_WIDTH + tx;
    float acc = 0.0f;
    
    for(int c = 0; c < C; c++){
        // copy data from global memory to shared memory
        if((h < H) && (w < W))
            shared_X[tIdx] = x4d(b, c, h, w);
        else
            shared_X[tIdx] = 0.0f;
        __syncthreads();

        // convolution
        if((h < H_out) && (w < W_out)){
            for(int p = 0; p < K; p++)
                for(int q = 0; q < K; q++)
                    acc += x4d(b, c, h + p, w + q) * k4d(m, c, p, q);
        }
        __syncthreads();
    }
    if((h < H_out) && (w < W_out))
        y4d(b, m, h, w) = acc;

#undef y4d
#undef x4d
#undef k4d
}


__host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_output, const float *host_input, const float *host_mask, float **device_output_ptr, float **device_input_ptr, float **device_mask_ptr, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // Allocate memory and copy over the relevant data structures to the GPU

    // We pass double pointers for you to initialize the relevant device pointers,
    //  which are passed to the other two functions.

    // Useful snippet for error checking
    // cudaError_t error = cudaGetLastError();
    // if(error != cudaSuccess)
    // {
    //     std::cout<<"CUDA error: "<<cudaGetErrorString(error)<<std::endl;
    //     exit(-1);
    // }
    const int H_out = Height - K + 1;
    const int W_out = Width - K + 1;
    int outputSize = Batch * Map_out * H_out * W_out;
    int inputSize = Batch * Channel * Height * Width;
    int kernelSize = Map_out * Channel * K * K;

    cudaMalloc((void **) device_output_ptr, outputSize * sizeof(float));
    cudaMalloc((void **) device_input_ptr, inputSize * sizeof(float));
    cudaMalloc((void **) device_mask_ptr, kernelSize * sizeof(float));
    cudaMemcpy(*device_input_ptr, host_input, inputSize * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(*device_mask_ptr, host_mask, kernelSize * sizeof(float), cudaMemcpyHostToDevice);
}


__host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // Set the kernel dimensions and call the kernel
    const int H_out = Height - K + 1;
    const int W_out = Width - K + 1;
    int W_grid = ceil(1.0 * (W_out + TILE_WIDTH - 1) / TILE_WIDTH); // Number of horizontal tiles for output maps
    int H_grid = ceil(1.0 * (H_out + TILE_WIDTH - 1) / TILE_WIDTH);  // Numer of vertical tiles for output maps
    int Y = W_grid * H_grid;

    int BLOCK_WIDTH = TILE_WIDTH + K - 1;

    dim3 blockDim(BLOCK_WIDTH, BLOCK_WIDTH, 1);
    dim3 gridDim(Map_out, Y, Batch);
    size_t shared_X_size = BLOCK_WIDTH * BLOCK_WIDTH * sizeof(float);
    conv_forward_kernel<<<gridDim, blockDim,shared_X_size>>>(device_output, device_input, device_mask, Batch, Map_out, Channel, Height, Width, K);
    cudaDeviceSynchronize();

}


__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input, float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    const int H_out = Height - K + 1;
    const int W_out = Width - K + 1;

    int outputSize = Batch * Map_out * H_out * W_out;

    // Copy the output back to host
    cudaMemcpy(host_output, device_output, outputSize * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(device_output);
    cudaFree(device_input);
    cudaFree(device_mask);

}


__host__ void GPUInterface::get_device_properties()
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    for(int dev = 0; dev < deviceCount; dev++)
    {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);

        std::cout<<"Device "<<dev<<" name: "<<deviceProp.name<<std::endl;
        std::cout<<"Computational capabilities: "<<deviceProp.major<<"."<<deviceProp.minor<<std::endl;
        std::cout<<"Max Global memory size: "<<deviceProp.totalGlobalMem<<std::endl;
        std::cout<<"Max Constant memory size: "<<deviceProp.totalConstMem<<std::endl;
        std::cout<<"Max Shared memory size per block: "<<deviceProp.sharedMemPerBlock<<std::endl;
        std::cout<<"Max threads per block: "<<deviceProp.maxThreadsPerBlock<<std::endl;
        std::cout<<"Max block dimensions: "<<deviceProp.maxThreadsDim[0]<<" x, "<<deviceProp.maxThreadsDim[1]<<" y, "<<deviceProp.maxThreadsDim[2]<<" z"<<std::endl;
        std::cout<<"Max grid dimensions: "<<deviceProp.maxGridSize[0]<<" x, "<<deviceProp.maxGridSize[1]<<" y, "<<deviceProp.maxGridSize[2]<<" z"<<std::endl;
        std::cout<<"Warp Size: "<<deviceProp.warpSize<<std::endl;
    }
}
