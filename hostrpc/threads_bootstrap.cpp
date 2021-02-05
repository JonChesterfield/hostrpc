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
#include "launch.hpp"

INCBIN(threads_bootstrap_so, "threads_bootstrap.gcn.so");

void run_threads_bootstrap(hsa::executable & ex, hsa_agent_t kernel_agent)
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


  fprintf(stderr, "Got kernarg block and a queue\n");
  
  uint64_t packet_id = hsa::acquire_available_packet_id(queue);
  hsa_kernel_dispatch_packet_t *packet =
      (hsa_kernel_dispatch_packet_t *)queue->base_address +
      (packet_id & (queue->size - 1));

  hsa::initialize_packet_defaults(packet);


  const char * kernel_entry = "__device_threads_bootstrap.kd";
  uint64_t device_threads_bootstrap = ex.get_symbol_address_by_name(kernel_entry);

  if (device_threads_bootstrap == 0) {
    fprintf(stderr, "kernel at %lu\n", device_threads_bootstrap);
    exit(1);
  }

 auto m = ex.get_kernel_info();
 
    auto it = m.find(std::string(kernel_entry));
  if (it != m.end())
    {
      packet->private_segment_size = it->second.private_segment_fixed_size;
      packet->group_segment_size = it->second.group_segment_fixed_size;
      fprintf(stderr,"setting sizes to %u/%u\n",       packet->private_segment_size,
             packet->group_segment_size);
    }
  else
    {
      fprintf(stderr, "Error: get_kernel_info failed\n");
      exit(1);
    }

  uint64_t z = 0;
  packet->kernel_object = device_threads_bootstrap;
  memcpy(&packet->kernarg_address, &z, 8);
  memcpy(&packet->completion_signal, &z, 8);

  // HSA marks this reserved, must be zero.
  // gfx9 passes the value through accurately, without error
  // will therefore use it as an implementation-defined arg slot
  packet->reserved2 = UINT64_MAX-1;
  

  packet_store_release((uint32_t *)packet,
                       header(HSA_PACKET_TYPE_KERNEL_DISPATCH),
                       kernel_dispatch_setup());

  hsa_signal_store_release(queue->doorbell_signal, packet_id);

  usleep(1000000);
}

int main_with_hsa()
{
  hsa_agent_t kernel_agent = hsa::find_a_gpu_or_exit();

  auto ex = hsa::executable(kernel_agent, threads_bootstrap_so_data,
                            threads_bootstrap_so_size);

  run_threads_bootstrap(ex, kernel_agent);

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
