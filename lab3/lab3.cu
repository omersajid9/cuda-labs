#include <wb.h>

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

#define TILE_WIDTH 16

// Compute C = A * B
__global__ void matrixMultiplyShared(float *A, float *B, float *C,
                                     int numARows, int numAColumns,
                                     int numBRows, int numBColumns,
                                     int numCRows, int numCColumns) {
  //@@ Insert code to implement matrix multiplication here
  //@@ You have to use shared memory for this MP
  __shared__ float tileA[TILE_WIDTH][TILE_WIDTH];
  __shared__ float tileB[TILE_WIDTH][TILE_WIDTH];

  int row = threadIdx.y + blockIdx.y * blockDim.y;
  int col = threadIdx.x + blockIdx.x * blockDim.x;

  int y = threadIdx.y;
  int x = threadIdx.x;

  float pValue = 0;

  for (int i = 0; i < (TILE_WIDTH + numBRows - 1) / TILE_WIDTH; i++) {
    // Phase 1: load the data
    // A_{row}{(TILE_WITH*i)+col} and B_{(TILE_WIDTH*i)+row}{col}
    if (row >= numARows || TILE_WIDTH*i + x >= numAColumns) {
      tileA[y][x] = 0.0f;
    } else {
      tileA[y][x] = A[row * numAColumns + TILE_WIDTH*i + x];
    }

    if (TILE_WIDTH*i + y >= numBRows || col >= numBColumns) {
      tileB[y][x] = 0.0f;
    } else {
      tileB[y][x] = B[(TILE_WIDTH*i + y) * numBColumns + col];
    }

    __syncthreads();
    
    // Phase 2: multiple data
    // tileA_{threadIdx.y * TILE_WIDTH + j} * tileA_{j * TILE_WIDTH + threadIdx.x}
    for (int j = 0; j < TILE_WIDTH; j++) {
      pValue += tileA[y][j] * tileB[j][x];
    }
    __syncthreads();
  }

  if (row < numCRows && col < numCColumns) {
      C[row * numCColumns + col] += pValue;
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
  //@@ Set numCRows and numCColumns
  numCRows = numARows;
  numCColumns = numBColumns;

  wbLog(TRACE, "The dimensions of A are ", numARows, " x ", numAColumns);
  wbLog(TRACE, "The dimensions of B are ", numBRows, " x ", numBColumns);
  wbLog(TRACE, "The dimensions of C are ", numCRows, " x ", numCColumns);


  int a_elems = numARows * numAColumns;
  size_t a_size = a_elems * sizeof(float);
  int b_elems = numBRows * numBColumns;
  size_t b_size = b_elems * sizeof(float);
  int c_elems = numCRows * numCColumns;
  size_t c_size = c_elems * sizeof(float);

  //@@ Allocate the hostC matrix
  hostC = (float*) malloc(c_size);

  //@@ Allocate GPU memory here
  float *deviceA;
  float *deviceB;
  float *deviceC;

  cudaMalloc((void**) &deviceA, a_size);
  cudaMalloc((void**) &deviceB, b_size);
  cudaMalloc((void**) &deviceC, c_size);

  //@@ Copy memory to the GPU here
  cudaMemcpy(deviceA, hostA, a_size, cudaMemcpyHostToDevice);
  cudaMemcpy(deviceB, hostB, b_size, cudaMemcpyHostToDevice);

  //@@ Initialize the grid and block dimensions here
  dim3 threadsPerBlock (TILE_WIDTH, TILE_WIDTH);
  dim3 numBlocks (std::ceil((float)numCColumns/(float)TILE_WIDTH), std::ceil((float)numCRows/(float)TILE_WIDTH));

  //@@ Launch the GPU Kernel here
  matrixMultiplyShared<<<numBlocks, threadsPerBlock>>>(deviceA, deviceB, deviceC, numARows, numAColumns, numBRows, numBColumns, numCRows, numCColumns);

  cudaError_t error = cudaDeviceSynchronize();
  wbCheck(error);

  //@@ Copy the GPU memory back to the CPU here
  cudaMemcpy(hostC, deviceC, c_size, cudaMemcpyDeviceToHost);

  //@@ Free the GPU memory here
  cudaFree(deviceA);
  cudaFree(deviceB);
  cudaFree(deviceC);

  wbSolution(args, hostC, numCRows, numCColumns);

  free(hostA);
  free(hostB);

  //@@ Free the hostC matrix
  free(hostC);
  return 0;
}
