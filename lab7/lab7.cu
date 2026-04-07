// MP Scan
// Given a list (lst) of length n
// Output its prefix sum = {lst[0], lst[0] + lst[1], lst[0] + lst[1] + ...
// +
// lst[n-1]}

#include <wb.h>

#define BLOCK_SIZE 512 //@@ You can change this
#define SECTION_SIZE (2 * BLOCK_SIZE)

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

__global__ void scan_kernel(float *input, float *output, float *blockSums, int len) {
  __shared__ float T[SECTION_SIZE];

  int start = 2 * blockIdx.x * BLOCK_SIZE;
  int tx = threadIdx.x;

  T[tx] = (start + tx < len) ? input[start + tx] : 0.0f;
  T[tx + BLOCK_SIZE] = (start + tx + BLOCK_SIZE < len) ? input[start + tx + BLOCK_SIZE] : 0.0f;
  __syncthreads();

  for (unsigned int stride = 1; stride <= BLOCK_SIZE; stride *= 2) {
    int index = (tx + 1) * stride * 2 - 1;
    if (index < SECTION_SIZE)
      T[index] += T[index - stride];
    __syncthreads();
  }

  for (unsigned int stride = SECTION_SIZE / 4; stride > 0; stride /= 2) {
    int index = (tx + 1) * stride * 2 - 1;
    if (index + stride < SECTION_SIZE)
      T[index + stride] += T[index];
    __syncthreads();
  }

  if (start + tx < len) output[start + tx] = T[tx];
  if (start + tx + BLOCK_SIZE < len) output[start + tx + BLOCK_SIZE] = T[tx + BLOCK_SIZE];

  if (blockSums != NULL && tx == 0)
    blockSums[blockIdx.x] = T[SECTION_SIZE - 1];
}

__global__ void add_kernel(float *output, float *scannedBlockSums, int len) {
  int start = 2 * blockIdx.x * BLOCK_SIZE;
  int tx = threadIdx.x;

  if (blockIdx.x > 0) {
    float val = scannedBlockSums[blockIdx.x - 1];
    if (start + tx < len)
      output[start + tx] += val;
    if (start + tx + BLOCK_SIZE < len)
      output[start + tx + BLOCK_SIZE] += val;
  }
}

int main(int argc, char **argv) {
  wbArg_t args;
  float *hostInput;  // The input 1D list
  float *hostOutput; // The output list
  float *deviceInput;
  float *deviceOutput;
  int numElements; // number of elements in the list

  args = wbArg_read(argc, argv);

  // Import data and create memory on host
  // The number of input elements in the input is numElements
  hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &numElements);
  hostOutput = (float *)malloc(numElements * sizeof(float));


  // Allocate GPU memory.
  wbCheck(cudaMalloc((void **)&deviceInput, numElements * sizeof(float)));
  wbCheck(cudaMalloc((void **)&deviceOutput, numElements * sizeof(float)));


  // Clear output memory.
  wbCheck(cudaMemset(deviceOutput, 0, numElements * sizeof(float)));

  // Copy input memory to the GPU.
  wbCheck(cudaMemcpy(deviceInput, hostInput, numElements * sizeof(float),
                     cudaMemcpyHostToDevice));

  //@@ Initialize the grid and block dimensions here
  int numBlocks = (numElements + SECTION_SIZE - 1) / SECTION_SIZE;

  //@@ Modify this to complete the functionality of the scan
  //@@ on the deivce
  float *aux1, *scannedAux1;
  wbCheck(cudaMalloc((void **) &aux1, numBlocks * sizeof(float)));
  wbCheck(cudaMalloc((void **) &scannedAux1, numBlocks * sizeof(float)));

  scan_kernel<<<numBlocks, BLOCK_SIZE>>>(deviceInput, deviceOutput, aux1, numElements);

  if (numBlocks > 1) {
    int numBlocks2 = (numBlocks + SECTION_SIZE - 1) / SECTION_SIZE;

    if (numBlocks2 == 1) {
      scan_kernel<<<1, BLOCK_SIZE>>>(aux1, scannedAux1, NULL, numBlocks);
    } else {
      float *aux2, *scannedAux2;
      wbCheck(cudaMalloc((void **) &aux2, numBlocks2 * sizeof(float)));
      wbCheck(cudaMalloc((void **) &scannedAux2, numBlocks2 * sizeof(float)));

      scan_kernel<<<numBlocks2, BLOCK_SIZE>>>(aux1, scannedAux1, aux2, numBlocks);
      scan_kernel<<<1, BLOCK_SIZE>>>(aux2, scannedAux2, NULL, numBlocks2);
      add_kernel<<<numBlocks2, BLOCK_SIZE>>>(scannedAux1, scannedAux2, numBlocks);

      cudaFree(aux2);
      cudaFree(scannedAux2);
    }

    add_kernel<<<numBlocks, BLOCK_SIZE>>>(deviceOutput, scannedAux1, numElements);
  }

  cudaDeviceSynchronize();

  // Copying output memory to the CPU
  wbCheck(cudaMemcpy(hostOutput, deviceOutput, numElements * sizeof(float),
                     cudaMemcpyDeviceToHost));


  //@@  Free GPU Memory
  cudaFree(deviceInput);
  cudaFree(deviceOutput);
  cudaFree(aux1);
  cudaFree(scannedAux1);

  wbSolution(args, hostOutput, numElements);

  free(hostInput);
  free(hostOutput);

  return 0;
}

