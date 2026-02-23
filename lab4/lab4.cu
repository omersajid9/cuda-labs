#include <wb.h>

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "CUDA error: ", cudaGetErrorString(err));              \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      return -1;                                                          \
    }                                                                     \
  } while (0)

//@@ Define any useful program-wide constants here
#define MASK_WIDTH 3
#define TILE_WIDTH 8

//@@ Define constant memory for device kernel here
__constant__ float deviceKernel [MASK_WIDTH][MASK_WIDTH][MASK_WIDTH];

__device__ int convert3Dto1DIndex(int x, int y, int z, int x_size, int y_size, int z_size) {
  return z * x_size * y_size + y * x_size + x;
}

__global__ void conv3d(float *input, float *output, const int z_size,
                       const int y_size, const int x_size) {
  //@@ Insert kernel code here

  // automatic variables
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int tz = threadIdx.z;

  int output_x = blockIdx.x * TILE_WIDTH + tx;
  int output_y = blockIdx.y * TILE_WIDTH + ty;
  int output_z = blockIdx.z * TILE_WIDTH + tz;

  int input_x = output_x - (MASK_WIDTH / 2);
  int input_y = output_y - (MASK_WIDTH / 2);
  int input_z = output_z - (MASK_WIDTH / 2);
  
  // shared tile memory
  __shared__ float tile[TILE_WIDTH + MASK_WIDTH - 1][TILE_WIDTH + MASK_WIDTH - 1][TILE_WIDTH + MASK_WIDTH - 1];


  // Load the tile data
  if (input_x >= 0 && input_x < x_size && input_y >= 0 && input_y < y_size && input_z >= 0 && input_z < z_size) {
    tile[tx][ty][tz] = input[convert3Dto1DIndex(input_x, input_y, input_z, x_size, y_size, z_size)];
  } else {
    tile[tx][ty][tz] = 0.0f;
  }
  __syncthreads();

  // compute convolution
  float pValue = 0.0f;
  if (tx < TILE_WIDTH && ty < TILE_WIDTH && tz < TILE_WIDTH) {
    for (int i = 0; i < MASK_WIDTH; i++) {
      for (int j = 0; j < MASK_WIDTH; j++) {
        for (int k = 0; k < MASK_WIDTH; k++) {
          pValue += tile[tx+i][ty+j][tz+k] * deviceKernel[i][j][k];
        }
      }
    }
    if (output_x >= 0 && output_x < x_size && output_y >= 0 && output_y < y_size && output_z >= 0 && output_z < z_size) {
      output[convert3Dto1DIndex(output_x, output_y, output_z, x_size, y_size, z_size)] = pValue;
    }
  }
}

int main(int argc, char *argv[]) {
  wbArg_t args;
  int z_size;
  int y_size;
  int x_size;
  int inputLength, kernelLength;
  float *hostInput;
  float *hostKernel;
  float *hostOutput;
  //@@ Initial deviceInput and deviceOutput here.

  float *deviceInput;
  float *deviceOutput;

  args = wbArg_read(argc, argv);

  // Import data
  hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &inputLength);
  hostKernel =
      (float *)wbImport(wbArg_getInputFile(args, 1), &kernelLength);
  hostOutput = (float *)malloc(inputLength * sizeof(float));

  // First three elements are the input dimensions
  z_size = hostInput[0];
  y_size = hostInput[1];
  x_size = hostInput[2];
  wbLog(TRACE, "The input size is ", z_size, "x", y_size, "x", x_size);
  assert(z_size * y_size * x_size == inputLength - 3);
  assert(kernelLength == 27); // Kernel is 3*3*3


  //@@ Allocate GPU memory here
  // Recall that inputLength is 3 elements longer than the input data
  // because the first  three elements were the dimensions

  int in_image_elems = z_size * y_size * x_size;
  size_t in_image_size = in_image_elems * sizeof(float);

  int out_image_elems = z_size * y_size * x_size;
  size_t out_image_size = out_image_elems * sizeof(float);

  cudaMalloc((void **) &deviceInput, in_image_size);
  cudaMalloc((void **) &deviceOutput, out_image_size);

  size_t kernel_size = kernelLength * sizeof(float);

  //@@ Copy input and kernel to GPU here
  // Recall that the first three elements of hostInput are dimensions and
  // do
  // not need to be copied to the gpu
  cudaMemcpy(deviceInput, hostInput + 3, in_image_size, cudaMemcpyHostToDevice);

  cudaMemcpyToSymbol(deviceKernel, hostKernel, kernel_size, 0, cudaMemcpyHostToDevice);

  //@@ Initialize grid and block dimensions here
  dim3 dimBlock(TILE_WIDTH + MASK_WIDTH - 1, TILE_WIDTH + MASK_WIDTH - 1, TILE_WIDTH + MASK_WIDTH - 1);
  dim3 dimGrid(ceil((float)x_size / TILE_WIDTH), ceil((float)y_size / TILE_WIDTH), ceil((float)z_size / TILE_WIDTH));

  //@@ Launch the GPU kernel here
  conv3d<<<dimGrid, dimBlock>>>(deviceInput, deviceOutput, z_size, y_size, x_size);
  cudaDeviceSynchronize();



  //@@ Copy the device memory back to the host here
  // Recall that the first three elements of the output are the dimensions
  // and should not be set here (they are set below)
  cudaMemcpy(hostOutput+3, deviceOutput, out_image_size, cudaMemcpyDeviceToHost);



  // Set the output dimensions for correctness checking
  hostOutput[0] = z_size;
  hostOutput[1] = y_size;
  hostOutput[2] = x_size;
  wbSolution(args, hostOutput, inputLength);

  //@@ Free device memory
  cudaFree(deviceInput);
  cudaFree(deviceOutput);


  // Free host memory
  free(hostInput);
  free(hostOutput);
  return 0;
}