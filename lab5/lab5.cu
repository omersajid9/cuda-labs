// Histogram Equalization

#include <wb.h>

#define HISTOGRAM_LENGTH 256
#define BLOCK_SIZE 16

//@@ insert code here

__host__ int ceil(int a, int b) {
  return (a + b - 1) / b;
}

__global__ void cast_float_to_char(float *input, unsigned char *output, int width, int height, int channels) {
  int w = blockDim.x * blockIdx.x + threadIdx.x;
  int h = blockDim.y * blockIdx.y + threadIdx.y;
  int c = threadIdx.z;

  if (w < width && h < height && c < channels) {
    output[h * width * channels + w * channels + c] = (unsigned char) (255 * input[h * width * channels + w * channels + c]);
  }
}

__global__ void cast_char_to_float(unsigned char *input, float *output, int width, int height, int channels) {
  int w = blockDim.x * blockIdx.x + threadIdx.x;
  int h = blockDim.y * blockIdx.y + threadIdx.y;
  int c = threadIdx.z;

  if (w < width && h < height && c < channels) {
    output[h * width * channels + w * channels + c] = (float) (input[h * width * channels + w * channels + c] / 255.0f);
  }
}

__global__ void color_to_grayscale(unsigned char *input, unsigned char *output, int width, int height, int channels) {
  int w = blockDim.x * blockIdx.x + threadIdx.x;
  int h = blockDim.y * blockIdx.y + threadIdx.y;

  if (w < width && h < height) {
    unsigned char r = input[h * width * channels + w * channels];
    unsigned char g = input[h * width * channels + w * channels + 1];
    unsigned char b = input[h * width * channels + w * channels + 2];

    unsigned char gs = (unsigned char) (0.21*r + 0.71*g + 0.07*b);
    output[h*width + w] = gs;
  }
}

__global__ void image_histogram(unsigned char *input, unsigned int *histogram, int width, int height) {
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  int w = blockDim.x * blockIdx.x + tx;
  int h = blockDim.y * blockIdx.y + ty;
  
  __shared__ unsigned int local_histogram[HISTOGRAM_LENGTH];
  
  int idx = ty * blockDim.x + tx;


  if (idx < HISTOGRAM_LENGTH) {
    local_histogram[idx] = 0;
  }
  __syncthreads();

  if (w < width && h < height) {
    unsigned char pixel = input[h * width + w];
    atomicAdd(&local_histogram[pixel], 1);
  }
  __syncthreads();

  if (idx < HISTOGRAM_LENGTH) {
    atomicAdd(&histogram[idx], local_histogram[idx]);
  }
  __syncthreads();
}

__host__ float p(unsigned int count, int width, int height) {
  return (float) count / (width * height);
}

__host__ void cdf_histogram(unsigned int *histogram, float *cdf, int width, int height) {
  cdf[0] = p(histogram[0], width, height);
  for (int i = 1; i < HISTOGRAM_LENGTH; i++) {
    cdf[i] = cdf[i-1] + p(histogram[i], width, height);
  }
}

__device__ float clamp(float x, float start, float end) {
  return fminf(fmaxf(x, start), end);
}

__device__ float correct_color(unsigned char val, float *cdf) {
  return clamp(255*(cdf[val] - cdf[0])/(1.0 - cdf[0]), 0, 255.0);
}

__global__ void correct_image(unsigned char *input, float *cdf, unsigned char *output, int width, int height, int channels) {
  int w = blockDim.x * blockIdx.x + threadIdx.x;
  int h = blockDim.y * blockIdx.y + threadIdx.y;
  int c = threadIdx.z;

  if (w < width && h < height && c < channels) {
    output[h * width * channels + w * channels + c] = (unsigned char) correct_color(input[h * width * channels + w * channels + c], cdf);
  }
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

  //Import data and create memory on host
  inputImage = wbImport(inputImageFile);
  imageWidth = wbImage_getWidth(inputImage);
  imageHeight = wbImage_getHeight(inputImage);
  imageChannels = wbImage_getChannels(inputImage);
  outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);


  // STEP 1: cast float to unsigned char

  // define host variables
  hostInputImageData = (float *) wbImage_getData(inputImage);

  float *inputImageFloat_d;
  unsigned char *inputImageChar_d;

  cudaMalloc((void **) &inputImageFloat_d, imageHeight * imageWidth * imageChannels *  sizeof(float));
  cudaMalloc((void **) &inputImageChar_d, imageHeight * imageWidth * imageChannels *  sizeof(unsigned char));

  cudaMemcpy(inputImageFloat_d, hostInputImageData, imageHeight * imageWidth * imageChannels * sizeof(float), cudaMemcpyHostToDevice);

  dim3 blockDim1(BLOCK_SIZE, BLOCK_SIZE, 3);
  dim3 gridDim1(ceil(imageWidth, BLOCK_SIZE), ceil(imageHeight, BLOCK_SIZE), 1);

  cast_float_to_char<<<gridDim1, blockDim1>>>(inputImageFloat_d, inputImageChar_d, imageWidth, imageHeight, imageChannels);
  cudaDeviceSynchronize();
  cudaFree(inputImageFloat_d);

  // STEP 2: convert rgb to grayscale
  unsigned char* grayscaleImageChar_d;

  cudaMalloc((void **) &grayscaleImageChar_d, imageHeight * imageWidth * sizeof(unsigned char));

  dim3 blockDim2(BLOCK_SIZE, BLOCK_SIZE, 1);
  dim3 gridDim2(ceil(imageWidth, BLOCK_SIZE), ceil(imageHeight, BLOCK_SIZE), 1);

  color_to_grayscale<<<gridDim2, blockDim2>>>(inputImageChar_d, grayscaleImageChar_d, imageWidth, imageHeight, imageChannels);

  // STEP 3: Compute histogram of grayscale
  unsigned int *histogram_h;
  unsigned int *histogram_d;

  histogram_h = (unsigned int *) malloc(HISTOGRAM_LENGTH * sizeof(unsigned int));
  cudaMalloc((void **) &histogram_d, HISTOGRAM_LENGTH * sizeof(unsigned int));
  cudaMemset(histogram_d, 0, HISTOGRAM_LENGTH * sizeof(unsigned int));

  dim3 blockDim3(BLOCK_SIZE, BLOCK_SIZE, 1);
  dim3 gridDim3(ceil(imageWidth, BLOCK_SIZE), ceil(imageHeight, BLOCK_SIZE), 1);

  image_histogram<<<gridDim3, blockDim3>>>(grayscaleImageChar_d, histogram_d, imageWidth, imageHeight);

  cudaDeviceSynchronize();
  cudaMemcpy(histogram_h, histogram_d, HISTOGRAM_LENGTH * sizeof(unsigned int), cudaMemcpyDeviceToHost);
  cudaFree(grayscaleImageChar_d);
  cudaFree(histogram_d);

  // STEP 4: Compute cdf of histogram (on CPU)
  float *cdf;
  cdf = (float *) malloc(HISTOGRAM_LENGTH * sizeof(float));

  cdf_histogram(histogram_h, cdf, imageWidth, imageHeight);

  float *cdf_d;
  cudaMalloc((void **) &cdf_d, HISTOGRAM_LENGTH * sizeof(float));
  cudaMemcpy(cdf_d, cdf, HISTOGRAM_LENGTH * sizeof(float), cudaMemcpyHostToDevice);

  // STEP 5: correct image color
  unsigned char *outputImageChar_d;
  cudaMalloc((void **) &outputImageChar_d, imageHeight * imageWidth * imageChannels * sizeof(unsigned char));

  dim3 blockDim5(BLOCK_SIZE, BLOCK_SIZE, 3);
  dim3 gridDim5(ceil(imageWidth, BLOCK_SIZE), ceil(imageHeight, BLOCK_SIZE), 1);

  correct_image<<<gridDim5, blockDim5>>>(inputImageChar_d, cdf_d, outputImageChar_d, imageWidth, imageHeight, imageChannels);
  cudaDeviceSynchronize();
  cudaFree(inputImageChar_d);
  cudaFree(cdf_d);

  // STEP 6: cast corrected image back to float
  float *outputImageFloat_d;
  cudaMalloc((void **) &outputImageFloat_d, imageHeight * imageWidth * imageChannels * sizeof(float));

  dim3 blockDim6(BLOCK_SIZE, BLOCK_SIZE, 3);
  dim3 gridDim6(ceil(imageWidth, BLOCK_SIZE), ceil(imageHeight, BLOCK_SIZE), 1);

  cast_char_to_float<<<gridDim6, blockDim6>>>(outputImageChar_d, outputImageFloat_d, imageWidth, imageHeight, imageChannels);
  cudaDeviceSynchronize();

  hostOutputImageData = (float *) wbImage_getData(outputImage);
  cudaMemcpy(hostOutputImageData, outputImageFloat_d, imageHeight * imageWidth * imageChannels * sizeof(float), cudaMemcpyDeviceToHost);
  cudaFree(outputImageChar_d);
  cudaFree(outputImageFloat_d);

  wbSolution(args, outputImage);

  free(histogram_h);
  free(cdf);

  return 0;
}

 