// Histogram Equalization

#include <wb.h>

#define HISTOGRAM_LENGTH 256
#define BLOCK_SIZE 256

//@@ insert code here

__global__ void floatToUnsign(float * deviceInputImageData, unsigned char * ucharImage, int inputImgSize)
{
  int tx = threadIdx.x;
  int bx = blockIdx.x;
  int i = tx + bx * blockDim.x;
  if (i < inputImgSize){
    ucharImage[i] = (unsigned char) (255 * deviceInputImageData[i]);
  }
  return;
}

__global__ void RgbToGray(unsigned char * ucharImage, unsigned char * grayImage, int imgSize)
{
  int tx = threadIdx.x;
  int bx = blockIdx.x;
  int i = tx + bx * blockDim.x;
  if (i < imgSize){
    unsigned char r = ucharImage[3 * i];
    unsigned char g = ucharImage[3 * i + 1];
    unsigned char b = ucharImage[3 * i + 2];
    grayImage[i] = (unsigned char) (0.21*r + 0.71*g + 0.07*b);
  }
  return;
}
__global__ void computeHistogram(unsigned char * grayImage, unsigned int * histogram, int imgSize)
{
  __shared__ unsigned int histogramShared[HISTOGRAM_LENGTH];
  int tx = threadIdx.x;
  int bx = blockIdx.x;
  int i = tx + bx * blockDim.x;
  int stride = BLOCK_SIZE;
  while (i < imgSize){
    atomicAdd(&histogramShared[grayImage[i]], 1);
    i += stride;
  }
  __syncthreads();
  histogram[tx] = histogramShared[tx];
  return;
}

__global__ void computeCDF(unsigned int * histogram,float * cdf, int imgSize)
{
  int tx = threadIdx.x;
  int bx = blockIdx.x;
  int idx = tx + bx * blockDim.x;
  if (idx == 0){
    unsigned int cum = 0;
    for (int i = 0; i < HISTOGRAM_LENGTH;i++){
      cum += histogram[i];
      cdf[i] = (float) cum / imgSize;
    }
  }
  return ;
}

// correctCDF<<<GridDim2,BlockDim2>>>(cdf);
__global__ void correctCDF(float * cdf)
{
  int tx = threadIdx.x;
  int bx = blockIdx.x;
  int idx = tx + bx * blockDim.x;
  if (idx == 0){
    float min = cdf[0];
    float max = 1.0;
    for (int i = 0; i < HISTOGRAM_LENGTH;i++){
      cdf[i] = (cdf[i] - min) / (max-min);
      cdf[i] *= 255.0;
      if (cdf[i] > 255.0){
        cdf[i] == 255.0;
      }
      if (cdf[i] < 0.0){
        cdf[i] = 0.0;
      }
    }
  }
  return ;
}
// applyHistogram<<<GridDim,BlockDim>>>(ucharImage,cdf,inputImgSize);
__global__ void applyHistogram(unsigned char * ucharImage, float * cdf, float * final_out_put_D, int inputImgSize)
{
  int tx = threadIdx.x;
  int bx = blockIdx.x;
  int i = tx + bx * blockDim.x;
  if (i < inputImgSize){
    final_out_put_D[i] = (float) cdf[ucharImage[i]] / 255.0;
  }
  return;
}

int main(int argc, char **argv) {
  wbArg_t args;
  int imageWidth;
  int imageHeight;
  int imageChannels;
  wbImage_t inputImage;
  wbImage_t outputImage;
  float *hostInputImageData;
  float *hostOutputImageData;
  const char *inputImageFile;

  //@@ Insert more code here

  args = wbArg_read(argc, argv); /* parse the input arguments */

  inputImageFile = wbArg_getInputFile(args, 0);

  wbTime_start(Generic, "Importing data and creating memory on host");
  inputImage = wbImport(inputImageFile);
  imageWidth = wbImage_getWidth(inputImage);
  imageHeight = wbImage_getHeight(inputImage);
  imageChannels = wbImage_getChannels(inputImage);
  outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);
  hostInputImageData = wbImage_getData(inputImage);
  hostOutputImageData = wbImage_getData(outputImage);
  wbTime_stop(Generic, "Importing data and creating memory on host");

  //@@ insert code here

  // Cast the image from float to unsigned char
  float *deviceInputImageData;
  unsigned char *ucharImage;
  unsigned int inputImgSize = imageWidth * imageHeight * imageChannels;
  cudaMalloc((void **) &deviceInputImageData, sizeof(float) * inputImgSize);
  cudaMalloc((void **) &ucharImage, sizeof(unsigned char) * inputImgSize);
  cudaMemcpy(deviceInputImageData,hostInputImageData,sizeof(float) * inputImgSize,cudaMemcpyHostToDevice);
  dim3 GridDim(ceil(1.0 * inputImgSize / BLOCK_SIZE),1,1);
  dim3 BlockDim(BLOCK_SIZE,1,1);
  floatToUnsign<<<GridDim,BlockDim>>>(deviceInputImageData,ucharImage,inputImgSize);
  cudaDeviceSynchronize();

  // Convert the image from RGB to GrayScale
  unsigned char *grayImage;
  unsigned int imgSize = imageWidth * imageHeight;
  cudaMalloc((void **) &grayImage, sizeof(unsigned char) * inputImgSize);
  dim3 GridDim1(ceil(1.0 * imgSize / BLOCK_SIZE),1,1);
  RgbToGray<<<GridDim1,BlockDim>>>(ucharImage,grayImage,imgSize);
  cudaDeviceSynchronize();

  // Compute the histogram of grayImage
  unsigned int *histogram;
  cudaMalloc((void **) &histogram, sizeof(unsigned int) * HISTOGRAM_LENGTH);
  dim3 GridDim2(1,1,1);
  computeHistogram<<<GridDim2,BlockDim>>>(grayImage,histogram,imgSize);
  cudaDeviceSynchronize();
  // Compute the Cumulative Distribution Function of histogram
  float *cdf;
  cudaMalloc((void **) &cdf, sizeof(float) * HISTOGRAM_LENGTH);
  dim3 BlockDim2(1,1,1);
  computeCDF<<<GridDim2,BlockDim2>>>(histogram,cdf,imgSize);
  cudaDeviceSynchronize();
  // Compute the minimum value of the CDF. The maximal value of the CDF should be 1.0. 
  correctCDF<<<GridDim2,BlockDim2>>>(cdf);

  // Apply the histogram equalization function
  float *final_out_put_D;
  cudaMalloc((void **) &final_out_put_D, sizeof(float) * inputImgSize);
  applyHistogram<<<GridDim,BlockDim>>>(ucharImage,cdf,final_out_put_D,inputImgSize);
  cudaDeviceSynchronize();

  cudaMemcpy(hostOutputImageData, final_out_put_D, sizeof(float) * inputImgSize, cudaMemcpyDeviceToHost);
  
  wbImage_setData(outputImage, hostOutputImageData);
  // free cuda
  cudaFree(deviceInputImageData);
  cudaFree(ucharImage);
  cudaFree(grayImage);
  cudaFree(histogram);
  cudaFree(cdf);
  

  wbSolution(args, outputImage);

  //@@ insert code here

  return 0;
}
