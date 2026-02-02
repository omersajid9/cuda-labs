#include <wb.h>
#include <cmath>

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)


// Compute C = A * B
__global__ void matrixMultiply(float *A, float *B, float *C, int numARows,
                               int numAColumns, int numBRows,
                               int numBColumns, int numCRows,
                               int numCColumns)
{
  //@@ Implement matrix multiplication kernel here
  int row = threadIdx.y + (blockDim.y * blockIdx.y);
  int col = threadIdx.x + (blockDim.x * blockIdx.x);

  // c_row_col = a_row_k * b_k_col
  if (col < numCColumns && row < numCRows) {
    float elemVal = 0;
    int K = numBRows;
    for (int k = 0; k < K; k++) {
      elemVal += A[row * numAColumns + k] * B[k * numBColumns + col];
    }
    C[row * numCColumns + col] = elemVal;
  }
}


int main(int argc, char **argv) {
  wbArg_t args;
  float *hostA; // The A matrix
  float *hostB; // The B matrix
  float *hostC; // The output C matrix

  int numARows;    // number of rows in the matrix A
  int numAColumns; // number of columns in the matrix A
  int numBRows;    // number of rows in the matrix B
  int numBColumns; // number of columns in the matrix B
  int numCRows;    // number of rows in the matrix C (you have to set this)
  int numCColumns; // number of columns in the matrix C (you have to set
                   // this)

  args = wbArg_read(argc, argv);

  //@@ Importing data and creating memory on host
  hostA = (float *)wbImport(wbArg_getInputFile(args, 0), &numARows,
                            &numAColumns);
  hostB = (float *)wbImport(wbArg_getInputFile(args, 1), &numBRows,
                            &numBColumns);
  wbLog(TRACE, "The dimensions of A are ", numARows, " x ", numAColumns);
  wbLog(TRACE, "The dimensions of B are ", numBRows, " x ", numBColumns);

  //@@ Set numCRows and numCColumns
  numCRows = numARows;
  numCColumns = numBColumns;
  wbLog(TRACE, "The dimensions of C are ", numCRows, " x ", numCColumns);

  //@@ Allocate the hostC matrix
  int cElems = numCRows * numCColumns;
  size_t cSize = cElems * sizeof(float);
  hostC = (float*) malloc(cSize);

  //@@ Allocate GPU memory here
  float *deviceA;
  float *deviceB;
  float *deviceC;

  cudaMalloc((void**) &deviceA, numARows * numAColumns * sizeof(float));
  cudaMalloc((void**) &deviceB, numBRows * numBColumns * sizeof(float));
  cudaMalloc((void**) &deviceC, numCRows * numCColumns * sizeof(float));

  //@@ Copy memory to the GPU here
  cudaMemcpy(deviceA, hostA, numARows * numAColumns * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(deviceB, hostB, numBRows * numBColumns * sizeof(float), cudaMemcpyHostToDevice);

  //@@ Initialize the grid and block dimensions here
  int BLOCK_SIZE = 16;
  dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
  dim3 blocks(std::ceil((float) numCColumns / (float) threadsPerBlock.x), std::ceil((float) numCRows / (float) threadsPerBlock.y));

  //@@ Launch the GPU Kernel here
  matrixMultiply<<<blocks, threadsPerBlock>>>(deviceA, deviceB, deviceC, numARows, numAColumns, numBRows, numBColumns, numCRows, numCColumns);

  cudaError_t error = cudaDeviceSynchronize();
  wbCheck(error);

  //@@ Copy the GPU memory back to the CPU here
  cudaMemcpy(hostC, deviceC, cSize, cudaMemcpyDeviceToHost);

  //@@ Free the GPU memory here
  cudaFree(deviceA);
  cudaFree(deviceB);
  cudaFree(deviceC);

  wbSolution(args, hostC, numCRows, numCColumns);

  free(hostA);
  free(hostB);
  //@@Free the hostC matrix
  free(hostC);

  return 0;
}

