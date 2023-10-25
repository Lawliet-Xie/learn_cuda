// index data
// z[i] = x[i] + y[i]
// for loop
// thread z[i]
// memory allocation
// memory copy  gpu_mem != cpu_mem
// kernel func
// memory copy

#include <stdio.h>
#include <math.h>



__global__ void vecAdd(const double *x, const double *y, double *z, int count)
{
    const int index = blockDim.x * blockIdx.x + threadIdx.x;
    // t00 t01 t02 t10 t11 t12 t20 t21 t22
    // t21: blockDim.x = 3, blockIdx.x = 2, threadIdx.x = 1 ==> index = 7
    if (index < count)
    {
        z[index] = x[index] + y[index];
    }
}


void vecAdd_cpu(const double *x, const double *y, double *z, int count)
{
    for (int i = 0; i < count; ++i)
    {
        z[i] = x[i] + y[i];
    }
}





// x[] + y[] = z[]
int main()
{
    const int N = 1000;
    const int M = sizeof(double) * N;

    // cpu mem alloc
    double *h_x = (double *) malloc(M);  // host
    double *h_y = (double *) malloc(M);
    double *h_z = (double *) malloc(M);
    double *cpu_res = (double *) malloc(M);

    // init
    for (int i = 0; i < N; ++i)
    {
        h_x[i] = 1;
        h_y[i] = 2;
    }

    // gpu mem alloc
    double *d_x, *d_y, *d_z;
    cudaMalloc((void **) &d_x, M);
    cudaMalloc((void **) &d_y, M);
    cudaMalloc((void **) &d_z, M);

    // cpu to gpu
    cudaMemcpy(d_x, h_x, M, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, M, cudaMemcpyHostToDevice);

    const int block_size = 128;
    const int grid_size = (N + block_size - 1) / block_size;

    // kernel func
    vecAdd<<<grid_size, block_size>>>(d_x, d_y, d_z, N);

    vecAdd_cpu(h_x, h_y, cpu_res, N);

    // gpu to cpu
    cudaMemcpy(h_z, d_z, M, cudaMemcpyDeviceToHost);

    bool error = false;
    for (int i = 0; i < N; ++i)
    {
        if (fabs(cpu_res[i] - h_z[i]) > (1.0e-10))
        {
            error = true;
        }
    }
    printf("Result: %s\n", error?"Error":"Pass");

    free(h_x);
    free(h_y);
    free(h_z);
    free(cpu_res);
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_z);


    return 0;
}



