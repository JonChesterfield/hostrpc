#include "hip/hip_runtime.h"

#include "detail/client_impl.hpp"
#include "detail/server_impl.hpp"

#include "allocator.hpp"
#include "host_client.hpp"

#include "x64_gcn_type.hpp"

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
