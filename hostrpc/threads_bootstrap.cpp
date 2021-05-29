#include "detail/platform_detect.hpp"

#if !defined(__OPENCL_C_VERSION__)
#include "detail/platform.hpp"
#include "pool_interface.hpp"
#endif

#if HOSTRPC_AMDGCN

#if defined(__OPENCL_C_VERSION__)

void pool_set_requested(void);
kernel void __device_pool_set_requested(void) { pool_set_requested(); }

void pool_bootstrap_target(void);
kernel void __device_pool_bootstrap_target(void) { pool_bootstrap_target(); }

void pool_bootstrap(void);
kernel void __device_pool_bootstrap_entry(void) { pool_bootstrap(); }

void pool_teardown(void);
kernel void __device_pool_teardown(void) { pool_teardown(); }

#else

struct example : public pool_interface::default_pool<example, 16>
{
  // example() {set_name(__func__); }
  void run()
  {
    if (platform::is_master_lane())
      printf("run from %u (of %u/%u)\n", get_current_uuid(), alive(),
             requested());
  }
};

extern "C"
{
  void pool_set_requested(void)
  {
    example::set_requested(pool_interface::load_from_reserved_addr());
  }

  void pool_bootstrap_target(void) { example::bootstrap_target(); }

  void pool_bootstrap(void)
  {
    __attribute__((visibility("default"))) extern hsa_packet::kernel_descriptor
        pool_bootstrap_target_desc asm("__device_pool_bootstrap_target.kd");
    example::instance()->bootstrap(
        pool_interface::load_from_reserved_addr(),
        (const unsigned char*)&pool_bootstrap_target_desc);
  }

  void pool_teardown(void) { example::teardown(); }
}
#endif
#endif

#if HOSTRPC_HOST
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

int async_kernel_set_requested(hsa::executable &ex, hsa_queue_t *queue,
                               uint32_t inline_argument)
{
  const char *name = "__device_pool_set_requested.kd";
  return hsa::launch_kernel(ex, queue, name, inline_argument, 0, {0});
}

static void wait_for_signal_equal_zero(hsa_signal_t signal,
                                       uint64_t timeout_hint = UINT64_MAX)
{
  do
    {
    }
  while (hsa_signal_wait_acquire(signal, HSA_SIGNAL_CONDITION_EQ, 0,
                                 timeout_hint, HSA_WAIT_STATE_ACTIVE) != 0);
}

static void run_threads_bootstrap_kernel(hsa::executable &ex,
                                         hsa_queue_t *queue,
                                         const char *bootstrap_entry_kernel)
{
  // Need to write to the first four bytes now, as this is the point that
  // knows what values to set

  // Kernels launched from the GPU, without reference to any host code,
  // presently all use this default header
  static_assert(
      hsa_packet::default_header ==
          hsa::packet_header(hsa::header(HSA_PACKET_TYPE_KERNEL_DISPATCH),
                             hsa::kernel_dispatch_setup()),
      "");
  ;

  // bootstrap invocation needs its own packet setup. The one it launches is
  // handled on the gpu side, without passing extra information via kernargs

  const char *kernel_entry = bootstrap_entry_kernel;
  uint64_t init_count = 1;

  hsa_signal_t signal;
  {
    auto rc = hsa_signal_create(1, 0, NULL, &signal);
    if (rc != HSA_STATUS_SUCCESS)
      {
        fprintf(stderr, "failed to create signal\n");
        exit(1);
      }
  }

  int rc = hsa::launch_kernel(ex, queue, kernel_entry, init_count, 0, signal);
  fprintf(stderr, "Launch kernel result %d\n", rc);

  // May no longer need a signal, since kernarg no longer in use
  wait_for_signal_equal_zero(signal);
  hsa_signal_destroy(signal);
}

int teardown_pool_kernel(hsa::executable &ex, hsa_queue_t *queue,
                         const char *name)
{
  // TODO: Make the signal earlier (and do other stuff) earlier, so this doesn't
  // fail
  hsa_signal_t signal;
  {
    auto rc = hsa_signal_create(1, 0, NULL, &signal);
    if (rc != HSA_STATUS_SUCCESS)
      {
        fprintf(stderr, "teardown: failed to create signal\n");
        return 1;
      }
  }

  // signal is in inline_argument, not in completion signal
  int rc = launch_kernel(ex, queue, name, signal.handle, 0, {0});

  if (rc != 0)
    {
      fprintf(stderr, "teardown: failed to launch kernel\n");
      hsa_signal_destroy(signal);
      return 1;
    }

  wait_for_signal_equal_zero(signal, 50000 /*000000*/);
  hsa_signal_destroy(signal);

  return 0;
}

// need to split enable print off from the macro
#undef printf
#include "hostrpc_printf.h"

int main_with_hsa()
{
  hsa_agent_t kernel_agent = hsa::find_a_gpu_or_exit();

  auto ex = hsa::executable(kernel_agent, threads_bootstrap_so_data,
                            threads_bootstrap_so_size);

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

  run_threads_bootstrap_kernel(ex, queue, "__device_pool_bootstrap_entry.kd");

  // leave them running for a while
  usleep(1000000);

  fprintf(stderr, "Start to wind down\n");
  int rc = teardown_pool_kernel(ex, queue, "__device_pool_teardown.kd");

  if (rc != 0)
    {
      fprintf(stderr, "teardown failed to start\n");
    }

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
