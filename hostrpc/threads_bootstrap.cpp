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
kernel void __device_pool_bootstrap(struct t data) { pool_bootstrap(data); }

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
  void run() { if (is_master_lane()) printf("run from %u (of %u/%u)\n", get_current_uuid(), alive(), requested()); }
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

  void pool_bootstrap_target(void)
  {
    example::bootstrap_target();
  }
  
  void * kernarg_segment_pointer()
  {
    // Quoting llc lowering of builtin_amdgcn_kernarg_segment_ptr,
    // if (!AMDGPU::isKernel(MF.getFunction().getCallingConv())) {
    //   This only makes sense to call in a kernel, so just lower to null.
    //   return DAG.getConstant(0, DL, VT);
    // }
    // which would be why it is returning null.

    __attribute__((address_space(4))) void* p = __builtin_amdgcn_dispatch_ptr();
    void * res;
    __builtin_memcpy(&res, (char*)p + (320/8), 8);
    return res;
  }
  
  void pool_bootstrap(struct t data /* data appears to be 64 bytes of zeros */)
  {
    // printf("&data %p\n", (const void*)&data);
    __attribute__((address_space(4))) void* ks =
      __builtin_amdgcn_kernarg_segment_ptr();

    void * kernarg_2 = kernarg_segment_pointer();
    
#if 0
    uint64_t w; __builtin_memcpy(&w, &ks, 8);
    printf("kernarg %p / %p, &data %p\n", (void*)ks, kernarg_2, &data);
    
    //    printf("dispatch %p\n", __builtin_amdgcn_dispatch_ptr());

    for (unsigned i = 0; i < 8; i++)
      {
        uint64_t tmp1;
        __builtin_memcpy(&tmp1, (char*)&data + 8*i, 8);
        uint64_t tmp2;
        __builtin_memcpy(&tmp2, (char*)kernarg_2 + 8*i, 8);
        
        printf("Arg[%u] 0x%lx 0x%lx\n", 8*8*i, tmp1, tmp2);
      }
#endif

    if (1) {
    example::instance()->bootstrap((const unsigned char *)&data);
    }    else  {
    example::instance()->bootstrap((const unsigned char *)kernarg_2);
    }    
  }

  void pool_teardown(void)
  {
    example::teardown();
  }
  
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

int teardown_pool(hsa::executable &ex, hsa_queue_t *queue)
{
  const char *name = "__device_pool_teardown.kd";

  // TODO: Make the signal earlier (and do other stuff) earlier, so this doesn't fail
  hsa_signal_t signal;
  {
  auto rc = hsa_signal_create(1, 0, NULL, &signal);
  if (rc != HSA_STATUS_SUCCESS) {
    fprintf(stderr,"teardown: failed to create signal\n");
    return 1;
  }
  }

  // signal is in inline_argument, not in completion signal
  int rc = launch_kernel(ex, queue, name, signal.handle, 0, {0});

  if (rc != 0) {
    fprintf(stderr,"teardown: failed to launch kernel\n");
    hsa_signal_destroy(signal);
    return 1;
  }
  
  do
    {
      // printf("waiting for teardown\n");

    } while (hsa_signal_wait_acquire(signal, 
                                 HSA_SIGNAL_CONDITION_EQ, 0, 50000 /*000000*/,
                                 HSA_WAIT_STATE_ACTIVE) != 0);

  hsa_signal_destroy(signal);
  
  return 0;
}

void run_threads_bootstrap(hsa::executable &ex, hsa_agent_t kernel_agent)
{
  hsa_queue_t *queue = hsa::create_queue(kernel_agent);
  if (!queue)
    {
      fprintf(stderr, "Failed to create queue\n");
      exit(1);
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
  if (init_packet(ex, "__device_pool_bootstrap_target.kd", kernarg) != 0)
    {
      fprintf(stderr, "init packet fail\n");
      exit(1);
    }

#if 0
  fprintf(stderr, "bootstrap target packet %lu:\n",(uint64_t)kernarg);
  dump_kernel((const unsigned char*)kernarg);
#endif
  // Need to write to the first four bytes now, as this is the point that
  // knows what values to set

  {
    uint32_t header = hsa::packet_header(hsa::header(HSA_PACKET_TYPE_KERNEL_DISPATCH),
                                         hsa::kernel_dispatch_setup());
    __builtin_memcpy((char*)kernarg, &header, 4);
  }

#if 0
  fprintf(stderr, "bootstrap target packet with header %lu:\n",(uint64_t)kernarg);
  dump_kernel((const unsigned char*)kernarg);
#endif
  
  // Sanity check we can launch the toplevel, it'll immediately return at
  // present
  if (0)
    {
      int rc =
          hsa::launch_kernel(ex, queue, "__device_pool_toplevel.kd", 0, 0, {0});
      printf("test run: %u\n", rc);

      // Also check the values written into kernarg will work as such if run
      // from here

      uint64_t packet_id = hsa::acquire_available_packet_id(queue);
      hsa_kernel_dispatch_packet_t *packet =
          (hsa_kernel_dispatch_packet_t *)queue->base_address +
          (packet_id & (queue->size - 1));

      memcpy(packet, kernarg, 64);

      hsa::packet_store_release((uint32_t *)packet,
                                hsa::header(HSA_PACKET_TYPE_KERNEL_DISPATCH),
                                hsa::kernel_dispatch_setup());
      hsa_signal_store_release(queue->doorbell_signal, packet_id);

      usleep(1000000);
      printf("Launched contents of kernarg\n");
    }

  fprintf(stderr, "Got kernarg block and a queue\n");

  // can asynchronously set the number of threads
  if (0)
    {
      int rc = async_kernel_set_requested(ex, queue, 1);
      if (rc == 0)
        {
          fprintf(stderr, "Launched set request\n");
        }
    }

  // bootstrap invocation needs it's own packet setup, but also needs a
  // 64 bit packet setup for the one it launches

  const char *kernel_entry = "__device_pool_bootstrap.kd";
  uint64_t init_count = 1;
  fprintf(stderr, "Launch! %lu\n", (uint64_t)kernarg);
  int rc = hsa::launch_kernel(ex, queue, kernel_entry, init_count,
                              (uint64_t)kernarg, {0});
  if (rc == 0)
    {
      fprintf(stderr, "Launched kernel\n");
    }
  else
    {
      fprintf(stderr, "Failed to launch, %d\n", rc);
    }

  usleep(1000000);

  fprintf(stderr, "Start to wind down\n");
  rc = teardown_pool(ex, queue);
  if (rc != 0)
    {
      fprintf(stderr, "teardown failed to start\n");
    }
  
  // Need a means of finding out whether alive has reached zero
  usleep(1000000);
  fprintf(stderr, "Wind down\n");

}

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
