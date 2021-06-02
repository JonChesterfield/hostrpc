#include "detail/platform_detect.hpp"

// TODO: Move the various macros into a header

// opencl kernel shims to work around __kernel being unavailable in c++
#if HOSTRPC_AMDGCN && defined(__OPENCL_C_VERSION__)
#define WRAP_VOID_IN_OPENCL_KERNEL(NAME) \
  void NAME(void);                       \
  kernel void __device_##NAME(void) { NAME(); }
#else
#define WRAP_VOID_IN_OPENCL_KERNEL(NAME)
#endif

#define GPU_OPENCL_WRAPPERS(SYMBOL)                     \
  WRAP_VOID_IN_OPENCL_KERNEL(SYMBOL##_set_requested)    \
  WRAP_VOID_IN_OPENCL_KERNEL(SYMBOL##_bootstrap_entry)  \
  WRAP_VOID_IN_OPENCL_KERNEL(SYMBOL##_bootstrap_target) \
  WRAP_VOID_IN_OPENCL_KERNEL(SYMBOL##_teardown)

// extern C functions that call into the type with run() implemented
#if HOSTRPC_AMDGCN && !defined(__OPENCL_C_VERSION__)
#define GPU_C_WRAPPERS(SYMBOL)                                           \
  extern "C"                                                             \
  {                                                                      \
    void SYMBOL##_set_requested(void)                                    \
    {                                                                    \
      SYMBOL::set_requested(pool_interface::load_from_reserved_addr());  \
    }                                                                    \
    void SYMBOL##_bootstrap_target(void) { SYMBOL::bootstrap_target(); } \
    void SYMBOL##_teardown(void) { SYMBOL::teardown(); }                 \
                                                                         \
    void SYMBOL##_bootstrap_entry(void)                                  \
    {                                                                    \
      __attribute__(                                                     \
          (visibility("default"))) extern hsa_packet::kernel_descriptor  \
          SYMBOL##_bootstrap_target_desc asm("__device_" #SYMBOL         \
                                             "_bootstrap_target.kd");    \
      SYMBOL::instance()->bootstrap(                                     \
          pool_interface::load_from_reserved_addr(),                     \
          (const unsigned char *)&SYMBOL##_bootstrap_target_desc);       \
    }                                                                    \
  }
#else
#define GPU_C_WRAPPERS(SYMBOL)
#endif

// functions that, post _initialize, run the corresponding function on the pool
#if HOSTRPC_HOST && !defined(__OPENCL_C_VERSION__)
#define HSA_KERNEL_WRAPPERS(SYMBOL)                                           \
                                                                              \
  std::array<gpu_symbols, maximum_number_gpu> SYMBOL##_global;                \
                                                                              \
  int SYMBOL##_initialize(hsa::executable &ex, hsa_queue_t *queue,            \
                          unsigned gpu)                                       \
  {                                                                           \
    if (gpu < maximum_number_gpu)                                             \
      {                                                                       \
        int rc = 0;                                                           \
        rc += initialize_kernel_info(ex,                                      \
                                     "__device_" #SYMBOL "_set_requested.kd", \
                                     &SYMBOL##_global[gpu].set_requested);    \
        rc += initialize_kernel_info(                                         \
            ex, "__device_" #SYMBOL "_bootstrap_entry.kd",                    \
            &SYMBOL##_global[gpu].bootstrap_entry);                           \
        rc += initialize_kernel_info(ex, "__device_" #SYMBOL "_teardown.kd",  \
                                     &SYMBOL##_global[gpu].teardown);         \
        if (rc != 0)                                                          \
          {                                                                   \
            return 1;                                                         \
          }                                                                   \
                                                                              \
        SYMBOL##_global[gpu].queue = queue;                                   \
        if (hsa_signal_create(1, 0, NULL, &SYMBOL##_global[gpu].signal) !=    \
            HSA_STATUS_SUCCESS)                                               \
          {                                                                   \
            return 1;                                                         \
          }                                                                   \
        return 0;                                                             \
      }                                                                       \
    return 1;                                                                 \
  }                                                                           \
                                                                              \
  void SYMBOL##_set_requested(unsigned gpu, uint64_t requested)               \
  {                                                                           \
    hsa_queue_t *queue = SYMBOL##_global[gpu].queue;                          \
    assert(gpu < maximum_number_gpu);                                         \
    gpu_kernel_info &req = SYMBOL##_global[gpu].set_requested;                \
    hsa::launch_kernel(req.symbol_address, req.private_segment_fixed_size,    \
                       req.group_segment_fixed_size, queue, requested, 0,     \
                       {0});                                                  \
  }                                                                           \
                                                                              \
  void SYMBOL##_bootstrap_entry(unsigned gpu, uint64_t requested)             \
  {                                                                           \
    hsa_queue_t *queue = SYMBOL##_global[gpu].queue;                          \
    assert(gpu < maximum_number_gpu);                                         \
    gpu_kernel_info &req = SYMBOL##_global[gpu].bootstrap_entry;              \
    hsa::launch_kernel(req.symbol_address, req.private_segment_fixed_size,    \
                       req.group_segment_fixed_size, queue, requested, 0,     \
                       {0});                                                  \
  }                                                                           \
                                                                              \
  void SYMBOL##_teardown(unsigned gpu)                                        \
  {                                                                           \
    hsa_queue_t *queue = SYMBOL##_global[gpu].queue;                          \
    assert(gpu < maximum_number_gpu);                                         \
    invoke_teardown(SYMBOL##_global[gpu].teardown,                            \
                    SYMBOL##_global[gpu].signal, queue, gpu);                 \
  }
#else
#define HSA_KERNEL_WRAPPERS(SYMBOL)
#endif

#if HOSTRPC_AMDGCN

#if !defined(__OPENCL_C_VERSION__)

#include "detail/platform.hpp"
#include "pool_interface.hpp"
struct example : public pool_interface::default_pool<example, 1024>
{
  void run()
  {
    if (platform::is_master_lane())
      printf("run from %u (of %u/%u)\n", get_current_uuid(), alive(),
             requested());

    platform::sleep_briefly();
  }
};

#endif
#endif

GPU_OPENCL_WRAPPERS(example);
GPU_C_WRAPPERS(example);
// todo: would like kernel wrappers instantiated here too

#if HOSTRPC_HOST && !defined(__OPENCL_C_VERSION__)
#include "hsa.hpp"
#include "incbin.h"
#include "launch.hpp"

INCBIN(threads_bootstrap_so, "threads_bootstrap.gcn.so");

struct gpu_kernel_info
{
  uint64_t symbol_address = 0;
  uint32_t private_segment_fixed_size = 0;
  uint32_t group_segment_fixed_size = 0;
};

struct gpu_symbols
{
  gpu_kernel_info set_requested;
  gpu_kernel_info bootstrap_entry;
  gpu_kernel_info teardown;
  hsa_signal_t signal = {0};
  hsa_queue_t *queue;
  ~gpu_symbols() { hsa_signal_destroy(signal); }
};

enum
{
  maximum_number_gpu = 4u,
};

namespace
{
inline int initialize_kernel_info(hsa::executable &ex, std::string name,
                                  gpu_kernel_info *info)
{
  uint64_t symbol_address = ex.get_symbol_address_by_name(name.c_str());
  auto m = ex.get_kernel_info();
  auto it = m.find(name);
  if (it == m.end() || symbol_address == 0)
    {
      return 1;
    }
  if ((it->second.private_segment_fixed_size > UINT32_MAX) ||
      (it->second.group_segment_fixed_size > UINT32_MAX))
    {
      return 1;
    }

  info->symbol_address = symbol_address;
  info->private_segment_fixed_size =
      (uint32_t)it->second.private_segment_fixed_size;
  info->group_segment_fixed_size =
      (uint32_t)it->second.group_segment_fixed_size;
  return 0;
}

inline void wait_for_signal_equal_zero(hsa_signal_t signal,
                                       uint64_t timeout_hint = UINT64_MAX)
{
  do
    {
    }
  while (hsa_signal_wait_acquire(signal, HSA_SIGNAL_CONDITION_EQ, 0,
                                 timeout_hint, HSA_WAIT_STATE_ACTIVE) != 0);
}

inline void invoke_teardown(gpu_kernel_info teardown, hsa_signal_t signal,
                            hsa_queue_t *queue, unsigned gpu)
{
  assert(gpu < maximum_number_gpu);

  const hsa_signal_value_t init = 1;
  hsa_signal_store_screlease(signal, init);

  hsa::launch_kernel(
      teardown.symbol_address, teardown.private_segment_fixed_size,
      teardown.group_segment_fixed_size, queue, signal.handle, 0, {0});

  wait_for_signal_equal_zero(signal, 50000 /*000000*/);
}
}  // namespace

// Kernels launched from the GPU, without reference to any host code,
// presently all use this default header

static_assert(
    hsa_packet::default_header ==
        hsa::packet_header(hsa::header(HSA_PACKET_TYPE_KERNEL_DISPATCH),
                           hsa::kernel_dispatch_setup()),
    "");

HSA_KERNEL_WRAPPERS(example)

// need to split enable print off from the macro
#undef printf
#include "hostrpc_printf.h"

int main_with_hsa()
{
  hsa_agent_t kernel_agent = hsa::find_a_gpu_or_exit();

  auto ex = hsa::executable(kernel_agent, threads_bootstrap_so_data,
                            threads_bootstrap_so_size);
  if (!ex.valid())
    {
      fprintf(stderr, "Failed to load executable %s\n",
              "threads_bootstrap.gcn.so");
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

  example_initialize(ex, queue, 0);

  example_bootstrap_entry(0, 8);

  // leave them running for a while
  usleep(1000000);

  fprintf(stderr, "Start to wind down\n");

  example_teardown(0);

  return 0;
}

int main()
{
  hsa::init state;
  fprintf(stderr, "In main\n");
  return main_with_hsa();
}

#endif
