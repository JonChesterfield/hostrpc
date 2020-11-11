#include "hip/hip_runtime.h"

#include "detail/client_impl.hpp"
#include "detail/server_impl.hpp"

#include "allocator.hpp"
#include "host_client.hpp"

#include "x64_gcn_type.hpp"

#include "hsa.hpp"

hostrpc::x64_gcn_type::client_type *client_instance = nullptr;
hostrpc::x64_gcn_type::server_type *server_instance = nullptr;

__device__ extern "C" int printf(const char *format, ...);

__global__ extern "C" void __device_start(int, char **, int *)
{
  printf("Hello world from hip\n");
}

int main()
{
  hsa::init with_hsa;

  hsa_agent_t kernel_agent = hsa::find_a_gpu_or_exit();

  hsa_region_t fine_grained_region = hsa::region_fine_grained(kernel_agent);
  hsa_region_t coarse_grained_region = hsa::region_coarse_grained(kernel_agent);
  {
    uint64_t fail = reinterpret_cast<uint64_t>(nullptr);
    if (fine_grained_region.handle == fail ||
        coarse_grained_region.handle == fail)
      {
        fprintf(stderr, "Failed to find allocation region on kernel agent\n");
        exit(1);
      }
  }

#if 1
  size_t N = 1920;
  hostrpc::x64_gcn_type p(N, fine_grained_region.handle,
                          coarse_grained_region.handle);
#endif

  __device_start<<<1, 1>>>(0, 0, 0);

  hipDeviceSynchronize();

  hipDeviceReset();

  return 0;
}
