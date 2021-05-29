#include "detail/platform_detect.hpp"

#if !defined(__OPENCL_C_VERSION__)
#include "pool_interface.hpp"
#endif

#if HOSTRPC_AMDGCN

struct t
{
  unsigned char data[64];
};

#if defined(__OPENCL_C_VERSION__)
void hsa_toplevel(void);
void hsa_set_requested(void);
void hsa_bootstrap_routine(void);

kernel void __device_threads_set_requested(void) { hsa_set_requested(); }
kernel void __device_threads_toplevel(void) { hsa_toplevel(); }

kernel void __device_threads_bootstrap(struct t a)
{
  (void)a;
  hsa_bootstrap_routine();
}

void pool_set_requested(void);
kernel void __device_pool_set_requested(void) { pool_set_requested(); }

void pool_bootstrap_target(void);
kernel void __device_pool_bootstrap_target(void) { pool_bootstrap_target(); }

void pool_bootstrap(struct t data);
kernel void __device_pool_bootstrap_entry(struct t data)
{
  pool_bootstrap(data);
}

void pool_teardown(void);
kernel void __device_pool_teardown(void) { pool_teardown(); }

#else

static inline uint32_t get_lane_id(void)
{
  return __builtin_amdgcn_mbcnt_hi(~0u, __builtin_amdgcn_mbcnt_lo(~0u, 0u));
}
static inline bool is_master_lane(void)
{
  // TODO: 32 wide wavefront, consider not using raw intrinsics here
  uint64_t activemask = __builtin_amdgcn_read_exec();

  // TODO: check codegen for trunc lowest_active vs expanding lane_id
  // TODO: ffs is lifted from openmp runtime, looks like it should be ctz
  uint32_t lowest_active = __builtin_ffsl(activemask) - 1;
  uint32_t lane_id = get_lane_id();

  // TODO: readfirstlane(lane_id) == lowest_active?
  return lane_id == lowest_active;
}

struct example : public pool_interface::default_pool<example, 16>
{
  //example() {set_name(__func__); }
  void run()
  {
    if (is_master_lane())
      printf("run from %u (of %u/%u)\n", get_current_uuid(), alive(),
             requested());
  }
};

__attribute__((always_inline)) static char* get_reserved_addr()
{
  __attribute__((address_space(4))) void* p = __builtin_amdgcn_dispatch_ptr();
  return (char*)p + 48;
}

static uint32_t load_from_reserved_addr()
{
  uint64_t tmp;
  __builtin_memcpy(&tmp, get_reserved_addr(), 8);
  return (uint32_t)tmp;
}

extern "C"
{
  void pool_set_requested(void)
  {
    example::set_requested(load_from_reserved_addr());
  }

  void pool_bootstrap_target(void) { example::bootstrap_target(); }

  void pool_bootstrap(struct t data /* data appears to be 64 bytes of zeros */)
  {
    example::instance()->bootstrap((const unsigned char*)&data);
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
                                         hsa_region_t kernarg_region,
                                         const char *bootstrap_entry_kernel,
                                         const char *bootstrap_target_kernel)
{
  auto kernarg_alloc = hsa::allocate(kernarg_region, 64);
  if (!kernarg_alloc)
    {
      fprintf(stderr, "Failed to allocate kernel arguments\n");
      exit(1);
    }
  hsa_kernel_dispatch_packet_t *kernarg =
      (hsa_kernel_dispatch_packet_t *)kernarg_alloc.get();
  if (init_packet(ex, bootstrap_target_kernel, kernarg) != 0)
    {
      fprintf(stderr, "init packet fail\n");
      exit(1);
    }

  fprintf(stderr, "target kernel:\n");
  dump_kernel((const unsigned char *)kernarg);
  
  // Need to write to the first four bytes now, as this is the point that
  // knows what values to set
  {
    uint32_t header =
        hsa::packet_header(hsa::header(HSA_PACKET_TYPE_KERNEL_DISPATCH),
                           hsa::kernel_dispatch_setup());
    __builtin_memcpy((char *)kernarg, &header, 4);
  }

  fprintf(stderr, "Got kernarg block and a queue\n");

  // bootstrap invocation needs it's own packet setup, but also needs a
  // 64 bit packet setup for the one it launches

  const char *kernel_entry = bootstrap_entry_kernel;
  uint64_t init_count = 1;
  fprintf(stderr, "Launch! %lu\n", (uint64_t)kernarg);

  hsa_signal_t signal;
  {
    auto rc = hsa_signal_create(1, 0, NULL, &signal);
    if (rc != HSA_STATUS_SUCCESS)
      {
        fprintf(stderr, "failed to create signal\n");
        exit(1);
      }
  }

  int rc = hsa::launch_kernel(ex, queue, kernel_entry, init_count,
                              (uint64_t)kernarg, signal);
  fprintf(stderr, "Launch kernel result %d\n", rc);

  // Kernarg needs to live until the kernel completes, so wait for the signal
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

#undef printf
#include "hostrpc_printf.h"

struct tbd
{
const char *bootstrap_entry_kernel;
  const char *bootstrap_target_kernel;
  const char * teardown_kernel;

  tbd()
  {
    fprintf(stderr, "%s %s %s\n", __func__, __FUNCTION__, __PRETTY_FUNCTION__);
  }

};

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

  hsa_region_t kernarg_region = hsa::region_kernarg(kernel_agent);

  tbd a;
  run_threads_bootstrap_kernel(ex, queue, kernarg_region,
                               "__device_pool_bootstrap_entry.kd",
                               "__device_pool_bootstrap_target.kd");

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
