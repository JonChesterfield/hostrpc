#define VISIBLE __attribute__((visibility("default")))

#if !defined __OPENCL__
#if defined(__AMDGCN__)
extern "C" void __device_persistent_call(void *)
{
  // Implemented by caller
}
#endif
#endif

// Kernel entry
#if defined __OPENCL__
void __device_persistent_kernel_cast(__global void *asargs);
kernel void __device_persistent_kernel(__global void *args)
{
  __device_persistent_kernel_cast(args);
}
#endif

#if !defined __OPENCL__

#include "queue_to_index.hpp"

#include "base_types.hpp"
#include "enqueue_dispatch.hpp"

#include <stddef.h>
#include <stdint.h>

#if defined(__x86_64__)
#include "test_common.hpp"  // round
#include "x64_host_x64_client.hpp"
#endif

#include "gcn_host_x64_client.hpp"

namespace hostrpc
{
// Lifecycle management is tricky for objects which are allocated on one system
// and copied to another, where they contain pointers into each other.
// One owning object is created. If successful, that can construct instances of
// a client or server class. These can be copied by memcpy, which is necessary
// to set up the instance across pcie. The client/server objects don't own
// memory so can be copied at will. They can be used until the owning instance
// destructs.

// Notes on the legality of the char state[] handling and aliasing.
// Constructing an instance into state[] is done with placement new, which needs
// the header <new> that is unavailable for amdgcn at present. Following
// libunwind's solution discussed at D57455, operator new is added as a member
// function to client_impl, server_impl. Combined with a reinterpret cast to
// select the right operator new, that creates the object. Access is via
// std::launder'ed reinterpret cast, but as one can't assume C++17 and doesn't
// have <new> for amdgcn, this uses __builtin_launder.

inline constexpr size_t client_counter_overhead()
{
  return client_counters::cc_total_count * sizeof(_Atomic(uint64_t));
}

inline constexpr size_t server_counter_overhead()
{
  return server_counters::sc_total_count * sizeof(_Atomic(uint64_t));
}

struct gcn_x64_t
{
#if defined(__x86_64__)
  static void copy_page(hostrpc::page_t *dst, hostrpc::page_t *src)
  {
    __builtin_memcpy(dst, src, sizeof(hostrpc::page_t));
  }
#endif

  struct fill
  {
    static void call(hostrpc::page_t *page, void *dv)
    {
#if defined(__x86_64__)
      hostrpc::page_t *d = static_cast<hostrpc::page_t *>(dv);
      copy_page(page, d);
#else
      (void)page;
      (void)dv;
#endif
    };
  };

  struct use
  {
    static void call(hostrpc::page_t *page, void *dv)
    {
#if defined(__x86_64__)
      hostrpc::page_t *d = static_cast<hostrpc::page_t *>(dv);
      copy_page(d, page);
#else
      (void)page;
      (void)dv;
#endif
    };
  };

#if defined(__AMDGCN__)
  static void gcn_server_callback(hostrpc::cacheline_t *)
  {
    // not yet implemented, maybe take a function pointer out of [0]
  }
#endif

  struct operate
  {
    static void call(hostrpc::page_t *page, void *)
    {
#if defined(__AMDGCN__)
      // Call through to a specific handler, one cache line per lane
      hostrpc::cacheline_t *l = &page->cacheline[platform::get_lane_id()];
      gcn_server_callback(l);
#else
      (void)page;
#endif
    };
  };

  struct clear
  {
    static void call(hostrpc::page_t *, void *) {}
  };

  using gcn_x64_type =
      gcn_x64_pair_T<hostrpc::size_runtime, fill, use, operate, clear>;

  gcn_x64_type instance;

  // for gfx906, probably want N = 2048
  gcn_x64_t(size_t N, uint64_t hsa_region_t_fine_handle,
            uint64_t hsa_region_t_coarse_handle)
      : instance(hostrpc::round(N), hsa_region_t_fine_handle,
                 hsa_region_t_coarse_handle)
  {
  }

  gcn_x64_t(const gcn_x64_t &) = delete;
  bool valid() { return true; }

  template <bool have_continuation>
  bool rpc_invoke(void *fill, void *use) noexcept
  {
    return instance.client.rpc_invoke<have_continuation>(fill, use);
  }

  bool rpc_handle(void *operate_state, void *clear_state,
                  uint32_t *location_arg) noexcept
  {
    return instance.server.rpc_handle(operate_state, clear_state, location_arg);
  }

  client_counters client_counters() { return instance.client.get_counters(); }
  server_counters server_counters() { return instance.server.get_counters(); }
};

}  // namespace hostrpc

#if 1
#if !defined(__AMDGCN__)
#include "amd_hsa_queue.h"
#include "hsa.h"
#include <stddef.h>
#include <stdint.h>

template <size_t expect, size_t actual>
static void assert_size_t_equal()
{
  static_assert(expect == actual, "");
}

void check_assumptions()
{
  assert_size_t_equal<sizeof(hsa_queue_t), 40>();
  assert_size_t_equal<sizeof(amd_queue_t), 256>();
  assert_size_t_equal<offsetof(hsa_queue_t, size), 24>();
  assert_size_t_equal<offsetof(hsa_queue_t, base_address), 8>();
  assert_size_t_equal<offsetof(hsa_queue_t, doorbell_signal), 16>();
  assert_size_t_equal<offsetof(amd_queue_t, write_dispatch_id), 56>();
  assert_size_t_equal<offsetof(amd_queue_t, read_dispatch_id), 128>();
}

#else
void check_assumptions() {}
#endif
#endif

#include <stddef.h>
#include <stdint.h>

#include "detail/platform.hpp"

struct kernel_args
{
  _Atomic(uint32_t) * control;
  void *application_args;
};

#if defined(__AMDGCN__)

// presentlin inlined into enqueue_dispatch
void kick_signal(uint64_t doorbell_handle)
{
  // uses a doorbell_handle to an amd_signal_t
  // that's a complex type, from which we need 'value' to hit the atomic
  // there's a uint32_t event_id and a uint64_t event_mailbox_ptr
  // try to get this working roughly first to avoid working out how to
  // link in the ockl stuff
  // see hsaqs.cl
  char *ptr = (char *)doorbell_handle;
  // kind is first 8 bytes, then a union containing value in next 8 bytes
  _Atomic(uint64_t) *event_mailbox_ptr = (_Atomic(uint64_t) *)(ptr + 16);
  uint32_t *event_id = (uint32_t *)(ptr + 24);

  assert(event_mailbox_ptr);  // I don't think this should be null

  if (platform::is_master_lane())
    {
      if (event_mailbox_ptr)
        {
          uint32_t id = *event_id;
          __opencl_atomic_store(event_mailbox_ptr, id, __ATOMIC_RELEASE,
                                __OPENCL_MEMORY_SCOPE_ALL_SVM_DEVICES);

          __builtin_amdgcn_s_sendmsg(1 | (0 << 4),
                                     __builtin_amdgcn_readfirstlane(id) & 0xff);
        }
    }
}

extern "C" void __device_persistent_kernel_call(_Atomic(uint32_t) * control,
                                                void *application_args)
{
  // The queue intrinsic returns a pointer to __constant memory. Mutating it
  // involves casting to, e.g., __global. The change may not be visible to this
  // kernel, despite the all_svm_devices scope. Should check codegen, given this
  // runs atomic fetch_add on it.

  // dispatch packet available in addr(4), __builtin_amdgcn_dispatch_ptr();
  if (*control == 0)
    {
      __opencl_atomic_store(control, 1, __ATOMIC_RELEASE,
                            __OPENCL_MEMORY_SCOPE_ALL_SVM_DEVICES);
      return;
    }
  else
    {
      __opencl_atomic_store(control, 2, __ATOMIC_RELEASE,
                            __OPENCL_MEMORY_SCOPE_ALL_SVM_DEVICES);
      return;
    }

  // TODO: Probably call this N times before the tail call
  // Could return based on the return value of this
  __device_persistent_call(application_args);

  enqueue_self();
}

extern "C" void __device_persistent_kernel_cast(
    __attribute__((address_space(1))) void *asargs)
{
  kernel_args *args = (kernel_args *)asargs;
  __device_persistent_kernel_call(args->control, args->application_args);
}

#endif

#if defined(__x86_64__)
#include "catch.hpp"
#include "incbin.h"
#include "launch.hpp"

INCBIN(persistent_kernel_so, "persistent_kernel.gcn.so");

TEST_CASE("persistent_kernel")
{
  hsa::init hsa;
  {
    using namespace hostrpc;

    hsa_agent_t kernel_agent = hsa::find_a_gpu_or_exit();
    auto ex = hsa::executable(kernel_agent, persistent_kernel_so_data,
                              persistent_kernel_so_size);
    CHECK(ex.valid());

    const char *kernel_entry = "__device_persistent_kernel.kd";
    uint64_t kernel_address = ex.get_symbol_address_by_name(kernel_entry);

    uint32_t kernel_private_segment_fixed_size = 0;
    uint32_t kernel_group_segment_fixed_size = 0;

    {
      auto m = ex.get_kernel_info();
      auto it = m.find(std::string(kernel_entry));
      if (it != m.end())
        {
          kernel_private_segment_fixed_size =
              it->second.private_segment_fixed_size;
          kernel_group_segment_fixed_size = it->second.group_segment_fixed_size;
        }
      else
        {
          printf("fatal: get_kernel_info failed\n");
          exit(1);
        }
    }

    printf("Kernel address %lx\n", kernel_address);

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

    hsa_region_t kernarg_region = hsa::region_kernarg(kernel_agent);
    hsa_region_t fine_grained_region = hsa::region_fine_grained(kernel_agent);
    hsa_region_t coarse_grained_region =
        hsa::region_coarse_grained(kernel_agent);

    {
      uint64_t fail = reinterpret_cast<uint64_t>(nullptr);
      if (kernarg_region.handle == fail || fine_grained_region.handle == fail ||
          coarse_grained_region.handle == fail)
        {
          fprintf(stderr, "Failed to find allocation region on kernel agent\n");
          exit(1);
        }
    }

    size_t N = 1920;
    hostrpc::gcn_x64_t p(N, fine_grained_region.handle,
                         coarse_grained_region.handle);

    // 4 bytes in its own page isn't ideal
    auto control_state =
        hsa::allocate(fine_grained_region, sizeof(_Atomic(uint32_t)));

    kernel_args example = {
        .control = new (control_state.get()) _Atomic(uint32_t),
        .application_args = 0,  // unused for now
    };
    __opencl_atomic_store(example.control, 0, __ATOMIC_RELEASE,
                          __OPENCL_MEMORY_SCOPE_ALL_SVM_DEVICES);

    auto l = launch_t<kernel_args>(kernel_agent, queue,
                                   /* kernel_address */ kernel_address,
                                   kernel_private_segment_fixed_size,
                                   kernel_group_segment_fixed_size,
                                   /* number waves */ 1, example);
    fprintf(stderr, "Launch instance\n");

    while (__opencl_atomic_load(example.control, __ATOMIC_ACQUIRE,
                                __OPENCL_MEMORY_SCOPE_ALL_SVM_DEVICES) == 0)
      {
        fprintf(stderr, "watching control\n");
        platform::sleep_noexcept(1000000);
      }

    fprintf(stderr, "Instance set control to non-zero\n");
  }
}

#endif

#endif
