
#include <stdio.h>

__global__ void cuda_hello() { printf("Hello world from cuda\n"); }

int main()
{
  cuda_hello<<<1, 1>>>();
  cudaDeviceSynchronize();

  cudaDeviceReset();
  return 0;
}
