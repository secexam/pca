## Procedure
```
!pip install git+https://github.com/andreinechaev/nvcc4jupyter.git
```
```
%load_ext nvcc4jupyter
```
```
then common.h la irukka code
then execute the question or given program
```
## Write a GPU based vector summation program using CUDA C. Find the execution configuration
```
// Type your device code here
__global__ void sumArraysOnGPU(float *A, float *B, float *C, const int N){
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i<N) C[i] = A[i] + B[i];
}

// set up device
int dev = 0;
cudaDeviceProp deviceProp;
CHECK(cudaGetDeviceProperties(&deviceProp, dev));
printf("Using Device %d: %s\n", dev, deviceProp.name);
CHECK(cudaSetDevice(dev));

// malloc device global memory
float *d_A, *d_B, *d_C;
CHECK(cudaMalloc((float**)&d_A, nBytes));
CHECK(cudaMalloc((float**)&d_B, nBytes));
CHECK(cudaMalloc((float**)&d_C, nBytes));

// transfer data from host to device
CHECK(cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice));
CHECK(cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice));
CHECK(cudaMemcpy(d_C, gpuRef, nBytes, cudaMemcpyHostToDevice));

// invoke kernel at host side
int iLen = 512;
dim3 block (iLen);
dim3 grid  ((nElem + block.x - 1) / block.x);

iStart = seconds();
sumArraysOnGPU<<<grid, block>>>(d_A, d_B, d_C, nElem);
CHECK(cudaDeviceSynchronize());
iElaps = seconds() - iStart;
printf("sumArraysOnGPU <<<  %d, %d  >>>  Time elapsed %f sec\n", grid.x,
        block.x, iElaps);

// check kernel error
CHECK(cudaGetLastError()) ;

// copy kernel result back to host side
CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));

// check device results
checkResult(hostRef, gpuRef, nElem);

// free device global memory
CHECK(cudaFree(d_A));
CHECK(cudaFree(d_B));
CHECK(cudaFree(d_C));

```
## Demonstrate the Matrix transposition on shared memory with grid (1,1) block (16,16).
```
CHECK(cudaMemset(d_C, 0, nBytes));
setRowReadRow<<<grid, block>>>(d_C);
CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));

CHECK(cudaMemset(d_C, 0, nBytes));
setColReadCol<<<grid, block>>>(d_C);
CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));

CHECK(cudaMemset(d_C, 0, nBytes));
setColReadCol2<<<grid, block>>>(d_C);
CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));

CHECK(cudaMemset(d_C, 0, nBytes));
setRowReadCol<<<grid, block>>>(d_C);
CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));

CHECK(cudaMemset(d_C, 0, nBytes));
setRowReadColDyn<<<grid, block, BDIMX*BDIMY*sizeof(int)>>>(d_C);
CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));

CHECK(cudaMemset(d_C, 0, nBytes));
setRowReadColPad<<<grid, block>>>(d_C);
CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));

CHECK(cudaMemset(d_C, 0, nBytes));
setRowReadColDynPad<<<grid, block, (BDIMX + IPAD)*BDIMY*sizeof(int)>>>(d_C);
CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));

```
## Implement sum reduction by unrolling8
```

// Kernel function declaration
__global__ void reduceUnrolling8(int *g_idata, int *g_odata, unsigned int n);

// Function to calculate elapsed time in milliseconds
double getElapsedTime(struct timeval start, struct timeval end)
{
    long seconds = end.tv_sec - start.tv_sec;
    long microseconds = end.tv_usec - start.tv_usec;
    double elapsed = seconds + microseconds / 1e6;
    return elapsed * 1000; // Convert to milliseconds
}

// Device memory allocation
int *d_idata, *d_odata;
cudaMalloc((void **)&d_idata, size);
cudaMalloc((void **)&d_odata, size);

// Copy input data from host to device
cudaMemcpy(d_idata, h_idata, size, cudaMemcpyHostToDevice);

// Launch the reduction kernel
reduceUnrolling8<<<gridSize, blockSize>>>(d_idata, d_odata, n);

// Copy the result from device to host
cudaMemcpy(h_odata, d_odata, size, cudaMemcpyDeviceToHost);

// Free Device Memory
cudaFree(d_idata);
cudaFree(d_odata);

```
## Implement sum reduction by unrolling16
```
// Kernel function declaration
__global__ void reduceUnrolling16(int *g_idata, int *g_odata, unsigned int n);

// Device memory allocation
int *d_idata, *d_odata;
cudaMalloc((void **)&d_idata, size);
cudaMalloc((void **)&d_odata, size);

// Copy input data from host to device
cudaMemcpy(d_idata, h_idata, size, cudaMemcpyHostToDevice);

// Define grid and block dimensions
dim3 blockSize(256); // 256 threads per block
dim3 gridSize((n + blockSize.x * 16 - 1) / (blockSize.x * 16));

// Compute the final sum on the GPU
int sum_gpu = 0;
for (unsigned int i = 0; i < gridSize.x; i++)
{
    sum_gpu += h_odata[i];
}

// Stop GPU timer
gettimeofday(&end_gpu, NULL);
double elapsedTime_gpu = getElapsedTime(start_gpu, end_gpu);

// Set thread ID
unsigned int tid = threadIdx.x;
unsigned int idx = blockIdx.x * blockDim.x * 16 + threadIdx.x;

// Convert global data pointer to the local pointer of this block
int *idata = g_idata + blockIdx.x * blockDim.x * 16;

// Unrolling 16
if (idx + 15 * blockDim.x < n)
{
        int a1 = g_idata[idx];
        int a2 = g_idata[idx + blockDim.x];
        int a3 = g_idata[idx + 2 * blockDim.x];
        int a4 = g_idata[idx + 3 * blockDim.x];
        int a5 = g_idata[idx + 4 * blockDim.x];
        int a6 = g_idata[idx + 5 * blockDim.x];
        int a7 = g_idata[idx + 6 * blockDim.x];
        int a8 = g_idata[idx + 7 * blockDim.x];
        int b1 = g_idata[idx + 8 * blockDim.x];
        int b2 = g_idata[idx + 9 * blockDim.x];
        int b3 = g_idata[idx + 10 * blockDim.x];
        int b4 = g_idata[idx + 11 * blockDim.x];
        int b5 = g_idata[idx + 12 * blockDim.x];
        int b6 = g_idata[idx + 13 * blockDim.x];
        int b7 = g_idata[idx + 14 * blockDim.x];
        int b8 = g_idata[idx + 15 * blockDim.x];
        g_idata[idx] = a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8 + b1 + b2 + b3 + b4 + b5 + b6 + b7 + b8;
}

```
## Write a CUDA C program for Matrix summation with a 2D grid and 2D blocks. Adapt it to integer matrix addition.
```
// grid 2D block 2D
__global__ void sumMatrixOnGPU2D(int *A, int *B, int *C, int NX, int NY)
{
    unsigned int ix = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int iy = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int idx = iy * NX + ix;

    if (ix < NX && iy < NY)
    {
        C[idx] = A[idx] + B[idx];
    }
}

// set up device
int dev = 0;
cudaDeviceProp deviceProp;
CHECK(cudaGetDeviceProperties(&deviceProp, dev));
printf("Using Device %d: %s\n", dev, deviceProp.name);
CHECK(cudaSetDevice(dev));

// set up data size of matrix
int nx = 1 << 14;
int ny = 1 << 14;

int nxy = nx * ny;
int nBytes = nxy * sizeof(float);
printf("Matrix size: nx %d ny %d\n", nx, ny);

// malloc device global memory
int *d_MatA, *d_MatB, *d_MatC;
CHECK(cudaMalloc((void **)&d_MatA, nBytes));
CHECK(cudaMalloc((void **)&d_MatB, nBytes));
CHECK(cudaMalloc((void **)&d_MatC, nBytes));

// transfer data from host to device
CHECK(cudaMemcpy(d_MatA, h_A, nBytes, cudaMemcpyHostToDevice));
CHECK(cudaMemcpy(d_MatB, h_B, nBytes, cudaMemcpyHostToDevice));

// invoke kernel at host side
int dimx = 32;
int dimy = 32;
dim3 block(dimx, dimy);
dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);

// check kernel error
CHECK(cudaGetLastError());

// copy kernel result back to host side
CHECK(cudaMemcpy(gpuRef, d_MatC, nBytes, cudaMemcpyDeviceToHost));

// check device results
checkResult(hostRef, gpuRef, nxy);

// free device global memory
CHECK(cudaFree(d_MatA));
CHECK(cudaFree(d_MatB));
CHECK(cudaFree(d_MatC));

```
## Write a CUDA C program to perform matrix addition with Unified memory.
```

```
## Write a CUDA C program to perform matrix addition with Unified memory.
```
// Define your program elements
#define SIZE 4
#define BLOCK_SIZE 2

// Kernel function to perform matrix multiplication
__global__ void matrixMultiply(int *a, int *b, int *c, int size)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    int sum = 0;
    for (int k = 0; k < size; ++k)
    {
        sum += a[row * size + k] * b[k * size + col];
    }
    c[row * size + col] = sum;
}

// Allocate memory on the device
cudaMalloc((void**)&dev_a, size);
cudaMalloc((void**)&dev_b, size);
cudaMalloc((void**)&dev_c, size);

// Copy input matrices from host to device memory
cudaMemcpy(dev_a, a, size, cudaMemcpyHostToDevice);
cudaMemcpy(dev_b, b, size, cudaMemcpyHostToDevice);

// Set grid and block sizes
dim3 dimGrid(SIZE / BLOCK_SIZE, SIZE / BLOCK_SIZE);
dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

// Launch kernel
matrixMultiply<<<dimGrid, dimBlock>>>(dev_a, dev_b, dev_c, SIZE);

// Copy result matrix from device to host memory
cudaMemcpy(c, dev_c, size, cudaMemcpyDeviceToHost);

// Print the elapsed time
printf("Elapsed Time: %.6f seconds\n", elapsed_time);

// Free device memory
cudaFree(dev_a);
cudaFree(dev_b);
cudaFree(dev_c);
```


