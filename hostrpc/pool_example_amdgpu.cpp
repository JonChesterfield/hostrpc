#include "detail/platform_detect.hpp"
#include "pool_interface.hpp"

POOL_INTERFACE_BOILERPLATE_AMDGPU(example, 32);

#if !defined(__OPENCL_C_VERSION__)
#include "detail/platform.hpp"

#if HOSTRPC_AMDGCN
uint32_t example::run(uint32_t state)
{
  if (0)
    if (platform::is_master_lane())
      printf("run %u from %u (of %u/%u)\n", state, get_current_uuid(), alive(),
             requested());

  platform::sleep_briefly();
  return state + 1;
}
#endif

#if HOSTRPC_HOST
#include "hsa.hpp"
#include "incbin.h"
#include "launch.hpp"

INCBIN(pool_example_amdgpu_so, "pool_example_amdgpu.gcn.so");

// need to split enable print off from the macro
#define WITH_PRINTF 0

#if WITH_PRINTF
#undef printf
#include "hostrpc_printf.h"
#endif

int main_with_hsa()
{
  hsa_agent_t kernel_agent = hsa::find_a_gpu_or_exit();
  fprintf(stderr, "Using agent %lu\n", kernel_agent.handle);
  auto ex = hsa::executable(kernel_agent, pool_example_amdgpu_so_data,
                            pool_example_amdgpu_so_size);
  if (!ex.valid())
    {
      fprintf(stderr, "Failed to load executable %s\n",
              "pool_example_amdgpu.gcn.so");
      exit(1);
    }
  else
    {
      fprintf(stderr, "Loaded executable %s\n", "pool_example_amdgpu.gcn.so");
    }

#if WITH_PRINTF
  if (hostrpc_print_enable_on_hsa_agent(ex, kernel_agent) != 0)
    {
      fprintf(stderr, "Failed to create host printf thread\n");
      exit(1);
    }
#endif
  
  hsa_queue_t *queue = hsa::create_queue(kernel_agent);
  if (!queue)
    {
      fprintf(stderr, "Failed to create queue\n");
      exit(1);
    }

  {
    unsigned char *base;
    __builtin_memcpy(&base, (char *)queue + 8, 8);
    uint32_t size;
    __builtin_memcpy(&size, (char *)queue + 24, 4);

    fprintf(stderr, "Queue is %p, packet base %p, size %u\n", queue, base,
            size);
  }

  example::initialize(ex, queue);

  example::bootstrap_entry(1024);

  // leave them running for a while
  usleep(10000000);

  fprintf(stderr, "Call teardown\n");

  example::teardown();

  example::finalize();


  {
    hsa_status_t rc = hsa_queue_destroy(queue);
    if (rc != HSA_STATUS_SUCCESS)
      {
        fprintf(stderr, "Failed to destroy queue: %s\n",
                hsa::status_string(rc));
      }
  }

  fprintf(stderr, "Finished\n");
  return 0;
}

int main()
{
  fprintf(stderr, "In main\n");
  hsa::init hsa_state;
  return main_with_hsa();
}

#endif
#endif
