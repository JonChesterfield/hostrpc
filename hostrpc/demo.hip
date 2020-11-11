
#include "hip/hip_runtime.h"

__device__ extern "C" int printf(const char *format, ...);

__global__ extern "C" void __device_start(int, char **, int *)
{
  printf("Hello world from hip\n");
}

int main()
{
  __device_start<<<1, 1>>>(0, 0, 0);

  hipDeviceSynchronize();

  hipDeviceReset();

  return 0;
}
