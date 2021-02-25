#include "detail/platform_detect.hpp"
#if HOSTRPC_AMDGCN
#if defined(__OPENCL_C_VERSION__)
void hsa_toplevel(void);
void hsa_set_requested(void);
void hsa_bootstrap_routine(void);

kernel void __device_threads_set_requested(void) { hsa_set_requested(); }
kernel void __device_threads_toplevel(void) { hsa_toplevel(); }
struct t
{
  unsigned char data[64];
};
kernel void __device_threads_bootstrap(struct t a)
{
  (void)a;
  hsa_bootstrap_routine();
}

#endif
#endif

#ifdef HOSTRPC_HOST
#if !defined(__OPENCL_C_VERSION__)
#include "hsa.hpp"
#include "incbin.h"
#include "launch.hpp"

INCBIN(threads_bootstrap_so, "threads_bootstrap.gcn.so");

int init_packet(hsa::executable &ex, const char *kernel_entry,
                hsa_kernel_dispatch_packet_t *packet)
{
  hsa::initialize_packet_defaults(packet);

  uint64_t symbol_address = ex.get_symbol_address_by_name(kernel_entry);
  auto m = ex.get_kernel_info();
  auto it = m.find(std::string(kernel_entry));
  if (it == m.end() || symbol_address == 0)
    {
      return 1;
    }

  packet->kernel_object = symbol_address;
  packet->private_segment_size = it->second.private_segment_fixed_size;
  packet->group_segment_size = it->second.group_segment_fixed_size;

  return 0;
}

// kernarg, signal may be zero
int launch_kernel(hsa::executable &ex, hsa_queue_t *queue,
                  const char *kernel_entry, uint64_t inline_argument,
                  uint64_t kernarg_address, uint64_t completion_signal)
{
  uint64_t packet_id = hsa::acquire_available_packet_id(queue);
  hsa_kernel_dispatch_packet_t *packet =
      (hsa_kernel_dispatch_packet_t *)queue->base_address +
      (packet_id & (queue->size - 1));

  if (init_packet(ex, kernel_entry, packet) != 0)
    {
      return 1;
    }

  memcpy(&packet->kernarg_address, &kernarg_address, 8);
  memcpy(&packet->completion_signal, &completion_signal, 8);

  // HSA marks this reserved, must be zero.
  // gfx9 passes the value through accurately, without error
  // will therefore use it as an implementation-defined arg slot
  memcpy(&packet->reserved2, &inline_argument, 8);

  packet_store_release((uint32_t *)packet,
                       header(HSA_PACKET_TYPE_KERNEL_DISPATCH),
                       kernel_dispatch_setup());

  hsa_signal_store_release(queue->doorbell_signal, packet_id);

  return 0;
}

int async_kernel_set_requested(hsa::executable &ex, hsa_queue_t *queue,
                               uint32_t inline_argument)
{
  const char *name = "__device_threads_set_requested.kd";
  return launch_kernel(ex, queue, name, inline_argument, 0, 0);
}

void run_threads_bootstrap(hsa::executable &ex, hsa_agent_t kernel_agent)
{
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

  // going to pass a packet as the argument, to relaunch on the device
  hsa_region_t kernarg_region = hsa::region_kernarg(kernel_agent);
  auto kernarg_alloc = hsa::allocate(kernarg_region, 64);
  if (!kernarg_alloc)
    {
      fprintf(stderr, "Failed to allocate kernel arguments\n");
      exit(1);
    }
  hsa_kernel_dispatch_packet_t *kernarg =
      (hsa_kernel_dispatch_packet_t *)kernarg_alloc.get();
  fprintf(stderr, "kernarg addr %p\n", kernarg);
  if (init_packet(ex, "__device_threads_toplevel.kd", kernarg) != 0)
    {
      fprintf(stderr, "init packet fail\n");
      exit(1);
    }

  // Sanity check we can launch the toplevel, it'll immediately return at present
  if (0) {
    int rc = launch_kernel(ex,
                           queue,
                           "__device_threads_toplevel.kd",
                           0, 0, 0);
    printf("test run: %u\n", rc);

    // Also check the values written into kernarg will work as such if run from here
    
    uint64_t packet_id = hsa::acquire_available_packet_id(queue);
    hsa_kernel_dispatch_packet_t *packet =
      (hsa_kernel_dispatch_packet_t *)queue->base_address +
      (packet_id & (queue->size - 1));

    memcpy(packet, kernarg, 64);

    packet_store_release((uint32_t *)packet,
                         header(HSA_PACKET_TYPE_KERNEL_DISPATCH),
                         kernel_dispatch_setup());
    hsa_signal_store_release(queue->doorbell_signal, packet_id);

    usleep(1000000);
    printf("Launched contents of kernarg\n");
    
  }

  
  fprintf(stderr, "Got kernarg block and a queue\n");

  // can asynchronously set the number of threads
  if (0) {
    int rc = async_kernel_set_requested(ex, queue, 1);
    if (rc == 0)
      {
        fprintf(stderr, "Launched set request\n");
      }
  }

  // bootstrap invocation needs it's own packet setup, but also needs a
  // 64 bit packet setup for the one it launches

  const char *kernel_entry = "__device_threads_bootstrap.kd";
  uint64_t init_count = 1;
  int rc =
      launch_kernel(ex, queue, kernel_entry, init_count, (uint64_t)kernarg, 0);
  if (rc == 0)
    {
      fprintf(stderr, "Launched kernel\n");
    }

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
