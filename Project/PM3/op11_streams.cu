#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"

#define TILE_WIDTH 23

__constant__ float const_k[5000];

__global__ void conv_forward_kernel(float *output, const float *input, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    /*
    Modify this function to implement the forward pass described in Chapter 16.
    We have added an additional dimension to the tensors to support an entire mini-batch
    The goal here is to be correct AND fast.

    Function paramter definitions:
    output - output
    input - input
    mask - convolution kernel
    Batch - batch_size (number of images in x)
    Map_out - number of output feature maps
    Channel - number of input feature maps
    Height - input height dimension
    Width - input width dimension
    K - kernel height and width (K x K)
    */

    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;
    const int W_grid = (Width_out + TILE_WIDTH - 1) / TILE_WIDTH;
    // (void)Height_out; // silence declared but never referenced warning. remove this line when you start working
    // (void)Width_out; // silence declared but never referenced warning. remove this line when you start working

    // We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    // An example use of these macros:
    // float a = y4d(0,0,0,0)
    // y4d(0,0,0,0) = a
#define out_4d(i3, i2, i1, i0) output[(i3) * (Map_out * Height_out * Width_out) + (i2) * (Height_out * Width_out) + (i1) * (Width_out) + i0]
#define in_4d(i3, i2, i1, i0) input[(i3) * (Channel * Height * Width) + (i2) * (Height * Width) + (i1) * (Width) + i0]
#define mask_4d(i3, i2, i1, i0) const_k[(i3) * (Channel * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    // Insert your GPU convolution kernel code here
    int m = blockIdx.x;
    int b = blockIdx.z;
    int h = (blockIdx.y / W_grid) * TILE_WIDTH + threadIdx.y;
    int w = (blockIdx.y % W_grid) * TILE_WIDTH + threadIdx.x;
    if((h < Height_out) && (w < Width_out)){
        float ans = 0.0f;
        for (int c = 0; c < Channel;c++){
                ans += in_4d(b,c,h+0,w+0) * mask_4d(m,c,0,0);
                ans += in_4d(b,c,h+0,w+1) * mask_4d(m,c,0,1);
                ans += in_4d(b,c,h+0,w+2) * mask_4d(m,c,0,2);
                ans += in_4d(b,c,h+0,w+3) * mask_4d(m,c,0,3);
                ans += in_4d(b,c,h+0,w+4) * mask_4d(m,c,0,4);
                ans += in_4d(b,c,h+0,w+5) * mask_4d(m,c,0,5);
                ans += in_4d(b,c,h+0,w+6) * mask_4d(m,c,0,6);

                ans += in_4d(b,c,h+1,w+0) * mask_4d(m,c,1,0);
                ans += in_4d(b,c,h+1,w+1) * mask_4d(m,c,1,1);
                ans += in_4d(b,c,h+1,w+2) * mask_4d(m,c,1,2);
                ans += in_4d(b,c,h+1,w+3) * mask_4d(m,c,1,3);
                ans += in_4d(b,c,h+1,w+4) * mask_4d(m,c,1,4);
                ans += in_4d(b,c,h+1,w+5) * mask_4d(m,c,1,5);
                ans += in_4d(b,c,h+1,w+6) * mask_4d(m,c,1,6);

                ans += in_4d(b,c,h+2,w+0) * mask_4d(m,c,2,0);
                ans += in_4d(b,c,h+2,w+1) * mask_4d(m,c,2,1);
                ans += in_4d(b,c,h+2,w+2) * mask_4d(m,c,2,2);
                ans += in_4d(b,c,h+2,w+3) * mask_4d(m,c,2,3);
                ans += in_4d(b,c,h+2,w+4) * mask_4d(m,c,2,4);
                ans += in_4d(b,c,h+2,w+5) * mask_4d(m,c,2,5);
                ans += in_4d(b,c,h+2,w+6) * mask_4d(m,c,2,6);  

                ans += in_4d(b,c,h+3,w+0) * mask_4d(m,c,3,0);
                ans += in_4d(b,c,h+3,w+1) * mask_4d(m,c,3,1);
                ans += in_4d(b,c,h+3,w+2) * mask_4d(m,c,3,2);
                ans += in_4d(b,c,h+3,w+3) * mask_4d(m,c,3,3);
                ans += in_4d(b,c,h+3,w+4) * mask_4d(m,c,3,4);
                ans += in_4d(b,c,h+3,w+5) * mask_4d(m,c,3,5);
                ans += in_4d(b,c,h+3,w+6) * mask_4d(m,c,3,6);

                ans += in_4d(b,c,h+4,w+0) * mask_4d(m,c,4,0);
                ans += in_4d(b,c,h+4,w+1) * mask_4d(m,c,4,1);
                ans += in_4d(b,c,h+4,w+2) * mask_4d(m,c,4,2);
                ans += in_4d(b,c,h+4,w+3) * mask_4d(m,c,4,3);
                ans += in_4d(b,c,h+4,w+4) * mask_4d(m,c,4,4);
                ans += in_4d(b,c,h+4,w+5) * mask_4d(m,c,4,5);
                ans += in_4d(b,c,h+4,w+6) * mask_4d(m,c,4,6);

                ans += in_4d(b,c,h+5,w+0) * mask_4d(m,c,5,0);
                ans += in_4d(b,c,h+5,w+1) * mask_4d(m,c,5,1);
                ans += in_4d(b,c,h+5,w+2) * mask_4d(m,c,5,2);
                ans += in_4d(b,c,h+5,w+3) * mask_4d(m,c,5,3);
                ans += in_4d(b,c,h+5,w+4) * mask_4d(m,c,5,4);
                ans += in_4d(b,c,h+5,w+5) * mask_4d(m,c,5,5);
                ans += in_4d(b,c,h+5,w+6) * mask_4d(m,c,5,6);

                ans += in_4d(b,c,h+6,w+0) * mask_4d(m,c,6,0);
                ans += in_4d(b,c,h+6,w+1) * mask_4d(m,c,6,1);
                ans += in_4d(b,c,h+6,w+2) * mask_4d(m,c,6,2);
                ans += in_4d(b,c,h+6,w+3) * mask_4d(m,c,6,3);
                ans += in_4d(b,c,h+6,w+4) * mask_4d(m,c,6,4);
                ans += in_4d(b,c,h+6,w+5) * mask_4d(m,c,6,5);
                ans += in_4d(b,c,h+6,w+6) * mask_4d(m,c,6,6);
        }
        out_4d(b, m, h, w) = ans;
    }

    #undef out_4d
    #undef in_4d
    #undef mask_4d
}

__host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_output, const float *host_input, const float *host_mask, float **device_output, float **device_input, float **device_mask_ptr, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    #define STREAM_NUM 20
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

    cudaMalloc((void **) device_output, outputSize * sizeof(float));
    cudaMalloc((void **) device_input, inputSize * sizeof(float));
    cudaMemcpyToSymbol(const_k, host_mask, kernelSize * sizeof(float));

    int W_grid = (W_out + TILE_WIDTH - 1) / TILE_WIDTH;
    int H_grid = (H_out + TILE_WIDTH - 1) / TILE_WIDTH;
    int Y = W_grid * H_grid;

    dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1);
    dim3 gridDim(Map_out, Y, Batch/STREAM_NUM);

    cudaStream_t stream[STREAM_NUM];
    for (int i = 0; i < STREAM_NUM; i++){
        cudaStreamCreate(&stream[i]);
    }   

    
    for (int i = 0; i < STREAM_NUM; i++){
        int x_offset = (inputSize / STREAM_NUM)* i;
        int y_offset = (outputSize / STREAM_NUM) * i;
        cudaMemcpyAsync((*device_input) + x_offset, host_input + x_offset, (inputSize / STREAM_NUM) * sizeof(float), cudaMemcpyHostToDevice, stream[i]);
        conv_forward_kernel<<<gridDim, blockDim, 0, stream[i]>>>((*device_output) + y_offset, (*device_input) + x_offset, Batch, Map_out, Channel, Height, Width, K);
        cudaMemcpyAsync((float*) host_output + y_offset, (*device_output) + y_offset, (outputSize / STREAM_NUM) * sizeof(float), cudaMemcpyDeviceToHost, stream[i]);
    }
    cudaDeviceSynchronize();

    for (int i = 0; i < STREAM_NUM; i++){
        cudaStreamDestroy(stream[i]);
    }
    // Free device memory
    cudaFree(device_output);
    cudaFree(device_input);

}

__host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // // Set the kernel dimensions and call the kernel
    // const int Height_out = Height - K + 1;
    // const int Width_out = Width - K + 1;
    // int W_grid = (Width_out + TILE_WIDTH - 1) / TILE_WIDTH;
    // int H_grid = (Height_out + TILE_WIDTH - 1) / TILE_WIDTH;
    // int Y = W_grid * H_grid;

    // dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1);
    // dim3 gridDim(Map_out, Y, Batch);
    // conv_forward_kernel<<<gridDim, blockDim>>>(device_output, device_input, device_mask, Batch, Map_out, Channel, Height, Width, K);
    // cudaDeviceSynchronize();

}

__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input, float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // const int H_out = Height - K + 1;
    // const int W_out = Width - K + 1;

    // int outputSize = Batch * Map_out * H_out * W_out;

    // // Copy the output back to host
    // cudaMemcpy(host_output, device_output, outputSize * sizeof(float), cudaMemcpyDeviceToHost);

    // // Free device memory
    // cudaFree(device_output);
    // cudaFree(device_input);

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
