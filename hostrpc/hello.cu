
#include <stdio.h>

__global__ extern "C" void __device_start(int, char**, int*)
{
  printf("Hello world from cuda\n");
}

int main()
{
  __device_start<<<1, 1>>>(0, 0, 0);
  cudaDeviceSynchronize();

  cudaDeviceReset();
  return 0;
}
