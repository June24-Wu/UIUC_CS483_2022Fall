// MP Scan
// Given a list (lst) of length n
// Output its prefix sum = {lst[0], lst[0] + lst[1], lst[0] + lst[1] + ...
// +
// lst[n-1]}

#include <wb.h>

#define BLOCK_SIZE 512 //@@ You can change this
#define SCAN 512

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

__global__ void scan(float *input, float *output, int len) {
  //@@ Modify the body of this function to complete the functionality of
  //@@ the scan on the device
  //@@ You may need multiple kernel calls; write your kernels before this
  //@@ function and call them from the host
 __shared__ float buffer[BLOCK_SIZE * 2];
  int tx = threadIdx.x;
  int bx = blockIdx.x;
  int index = bx * BLOCK_SIZE + tx;
  int dataIndex = bx * BLOCK_SIZE * 2 + tx;
  
  // load data 
  if (dataIndex >= len) buffer[tx] = 0;
  else  buffer[tx] = input[dataIndex];
  if (dataIndex + BLOCK_SIZE >= len) buffer[tx + BLOCK_SIZE] = 0;
  else buffer[tx + BLOCK_SIZE] = input[dataIndex + BLOCK_SIZE];

  __syncthreads();
  
  int stride = 1;
  while (stride < BLOCK_SIZE * 2)  // stride less than 
  {
    // forward reduction tree, first round: 1, 3, 5,7 | second round: 3, 7
    int idx = (tx + 1) * stride * 2 - 1;
    if (idx < 2 * BLOCK_SIZE && idx - stride >= 0)
      buffer[idx] += buffer[idx - stride];
    stride *= 2; // stride doubled since every two nodes reduces to one 
    __syncthreads();
  }
  
  stride = BLOCK_SIZE / 2;
  while (stride > 0)
  {
    int idx = (tx + 1) * stride * 2 - 1;
    if (idx + stride < 2 * BLOCK_SIZE)
       buffer[idx + stride] += buffer[idx]; // idx + stride == 2 * BLOCK_SIZE, edge
    stride /= 2;
    __syncthreads();
  }
  
  // write out the elements
  if (dataIndex < len) output[dataIndex] = buffer[tx];
  if (dataIndex + BLOCK_SIZE < len) output[dataIndex + BLOCK_SIZE] = buffer[tx + BLOCK_SIZE];
  
  return;
}

int main(int argc, char **argv) {
  wbArg_t args;
  float *hostInput;  // The input 1D list
  float *hostOutput; // The output list
  float *deviceInput;
  float *deviceOutput;
  int numElements; // number of elements in the list

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &numElements);
  hostOutput = (float *)malloc(numElements * sizeof(float));
  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The number of input elements in the input is ",
        numElements);

  wbTime_start(GPU, "Allocating GPU memory.");
  wbCheck(cudaMalloc((void **)&deviceInput, numElements * sizeof(float)));
  wbCheck(cudaMalloc((void **)&deviceOutput, numElements * sizeof(float)));
  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Clearing output memory.");
  wbCheck(cudaMemset(deviceOutput, 0, numElements * sizeof(float)));
  wbTime_stop(GPU, "Clearing output memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");
  wbCheck(cudaMemcpy(deviceInput, hostInput, numElements * sizeof(float),
                     cudaMemcpyHostToDevice));
  wbTime_stop(GPU, "Copying input memory to the GPU.");

  //@@ Initialize the grid and block dimensions here
  int gd = ceil(numElements/(BLOCK_SIZE * 1.0));
  dim3 grid(gd, 1, 1);
  dim3 block(BLOCK_SIZE, 1, 1);

  wbTime_start(Compute, "Performing CUDA computation");
  //@@ Modify this to complete the functionality of the scan
  //@@ on the deivce
  for(int j = 0; j < gd; j++) {
        int t = j * BLOCK_SIZE;
        if(j != 0) {
                float temp = 0;
                cudaMemcpy(&temp, &deviceOutput[t - 1], sizeof(float), cudaMemcpyDeviceToHost);
                temp += hostInput[t];
                cudaMemcpy(&deviceInput[t], &temp, sizeof(float),cudaMemcpyHostToDevice);
        }
        scan<<<grid, block>>>(&deviceInput[t], &deviceOutput[t], numElements);
  }

  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
  wbCheck(cudaMemcpy(hostOutput, deviceOutput, numElements * sizeof(float),
                     cudaMemcpyDeviceToHost));
  wbTime_stop(Copy, "Copying output memory to the CPU");

  wbTime_start(GPU, "Freeing GPU Memory");
  cudaFree(deviceInput);
  cudaFree(deviceOutput);
  wbTime_stop(GPU, "Freeing GPU Memory");

  wbSolution(args, hostOutput, numElements);

  free(hostInput);
  free(hostOutput);

  return 0;
}





