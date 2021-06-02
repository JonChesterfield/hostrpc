#include "detail/platform_detect.hpp"

#include "pool_interface.hpp"

POOL_INTERFACE_BOILERPLATE_AMDGPU(example, 32);

#if !defined(__OPENCL_C_VERSION__)
#include "detail/platform.hpp"

#if HOSTRPC_AMDGCN

void example::run()
{
  if (platform::is_master_lane())
    printf("run from %u (of %u/%u)\n", get_current_uuid(), alive(),
           requested());

  platform::sleep_briefly();
}

#endif

#if HOSTRPC_HOST
#include "hsa.hpp"
#include "incbin.h"
#include "launch.hpp"

INCBIN(pool_example_amdgpu_so, "pool_example_amdgpu.gcn.so");

// need to split enable print off from the macro
#undef printf
#include "hostrpc_printf.h"

int main_with_hsa()
{
  hsa_agent_t kernel_agent = hsa::find_a_gpu_or_exit();

  auto ex = hsa::executable(kernel_agent, pool_example_amdgpu_so_data,
                            pool_example_amdgpu_so_size);
  if (!ex.valid())
    {
      fprintf(stderr, "Failed to load executable %s\n",
              "pool_example_amdgpu.gcn.so");
      exit(1);
    }

  if (hostrpc_print_enable_on_hsa_agent(ex, kernel_agent) != 0)
    {
      fprintf(stderr, "Failed to create host printf thread\n");
      exit(1);
    }

  hsa_queue_t *queue = hsa::create_queue(kernel_agent);
  if (!queue)
    {
      fprintf(stderr, "Failed to create queue\n");
      exit(1);
    }

  example_initialize(ex, queue);

  example_bootstrap_entry(8);

  // leave them running for a while
  usleep(1000000);

  fprintf(stderr, "Start to wind down\n");

  example_teardown();

  example_finalize();

  return 0;
}

int main()
{
  hsa::init state;
  fprintf(stderr, "In main\n");
  return main_with_hsa();
}

#endif
#endif
