#include "detail/platform_detect.hpp"

#if HOSTRPC_AMDGCN
#if defined(__OPENCL_C_VERSION__)

void hsa_start_routine();

kernel void __device_threads_bootstrap(__global char *args)
{
  (void)args;
  hsa_start_routine();
}

#endif
#endif

#ifdef HOSTRPC_HOST
#if !defined(__OPENCL_C_VERSION__)
#include "hsa.hpp"
#include "incbin.h"

INCBIN(threads_bootstrap_so, "threads_bootstrap.gcn.so");

void run_threads_bootstrap(hsa_agent_t kernel_agent)
{
  hsa_region_t kernarg_region = hsa::region_kernarg(kernel_agent);
  constexpr uint32_t max_threads = 4096;

  // Say maximum threads will be 4096
  // Kernarg region minimum alignment 16 bytes
  // Means 16 pages
  // To handle it via alignment tricks on the device, needs to be ~32 pages

  auto kernarg_alloc = hsa::allocate(kernarg_region, max_threads * 16 * 2);
  if (!kernarg_alloc)
    {
      fprintf(stderr, "Failed to allocate %u bytes for kernel arguments\n",
              max_threads * 16 * 2);
      exit(1);
    }

  char *kernarg = (char *)kernarg_alloc.get();
  fprintf(stderr, "kernarg addr %p\n", kernarg);

  hsa_queue_t *queue;
  {
    hsa_status_t rc = hsa_queue_create(
        kernel_agent /* make the queue on this agent */,
        131072 /* todo: size it, this hardcodes max size for vega20 */,
        HSA_QUEUE_TYPE_MULTI /* baseline */,
        NULL /* called on every async event? */,
        NULL /* data passed to previous */,
        // If sizes exceed these values, things are supposed to work slowly
        UINT32_MAX /* private_segment_size, 32_MAX is unknown */,
        UINT32_MAX /* group segment size, as above */, &queue);
    if (rc != HSA_STATUS_SUCCESS)
      {
        fprintf(stderr, "Failed to create queue\n");
        exit(1);
      }
  }
}

int main_with_hsa()
{
  hsa_agent_t kernel_agent = hsa::find_a_gpu_or_exit();

  auto ex = hsa::executable(kernel_agent, threads_bootstrap_so_data,
                            threads_bootstrap_so_size);

  run_threads_bootstrap(kernel_agent);

  return 0;
}

int main()
{
  hsa::init state;
  return main_with_hsa();
}

#endif
#endif
