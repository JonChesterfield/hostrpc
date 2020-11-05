#define VISIBLE __attribute__((visibility("default")))

#include "cxa_atexit.hpp"

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
  return client_counters::cc_total_count * sizeof(HOSTRPC_ATOMIC(uint64_t));
}

inline constexpr size_t server_counter_overhead()
{
  return server_counters::sc_total_count * sizeof(HOSTRPC_ATOMIC(uint64_t));
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
  static void gcn_server_callback(hostrpc::cacheline_t *line)
  {
    // not yet implemented, maybe take a function pointer out of [0]
    uint64_t l01 = line->element[0] * line->element[1];
    uint64_t l23 = line->element[2] * line->element[3];
    line->element[0] = l01 * l23;
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
    static void call(hostrpc::page_t *page, void *)
    {
#if defined(__AMDGCN__)
      hostrpc::cacheline_t *l = &page->cacheline[platform::get_lane_id()];
      for (unsigned i = 0; i < 8; i++)
        {
          l->element[i] = 0;
        }
#else
      (void)page;
#endif
    }
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
  HOSTRPC_ATOMIC(uint32_t) * control;
  void *application_args;
};

#if defined(__AMDGCN__)

__attribute__((loader_uninitialized))
VISIBLE hostrpc::gcn_x64_t::gcn_x64_type::server_type server_instance[1];

uint32_t cas_fetch_dec(HOSTRPC_ATOMIC(uint32_t) * addr)
{
  uint32_t current =
      platform::atomic_load<uint32_t, __ATOMIC_RELAXED,
                            __OPENCL_MEMORY_SCOPE_ALL_SVM_DEVICES>(addr);
  while (1)
    {
      uint32_t replace = current - 1;

      bool r = __opencl_atomic_compare_exchange_weak(
          addr, &current, replace, __ATOMIC_ACQ_REL, __ATOMIC_ACQ_REL,
          __OPENCL_MEMORY_SCOPE_ALL_SVM_DEVICES);

      if (r)
        {
          return current;
        }
    }
}

extern "C" void __device_persistent_kernel_call(HOSTRPC_ATOMIC(uint32_t) *
                                                    control,
                                                void *application_args)
{
  // The queue intrinsic returns a pointer to __constant memory. Mutating it
  // involves casting to, e.g., __global. The change may not be visible to this
  // kernel, despite the all_svm_devices scope. Should check codegen, given this
  // runs atomic fetch_add on it.

  // dispatch packet available in addr(4), __builtin_amdgcn_dispatch_ptr();
  (void)application_args;

  uint32_t location_arg = 0;

  if (server_instance[0].rpc_handle(0, 0, &location_arg))
    {
      uint32_t todo = platform::critical<uint32_t>(
          [&]() { return cas_fetch_dec(control); });

      if (todo == 1)
        {
          return;
        }
    }

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

    uint64_t server_address = ex.get_symbol_address_by_name("server_instance");

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

    {
      auto c =
          hsa::allocate(fine_grained_region,
                        sizeof(hostrpc::gcn_x64_t::gcn_x64_type::server_type));
      void *vc = c.get();
      memcpy(vc, &p.instance.server, sizeof(p.instance.server));

      int rc = hsa::copy_host_to_gpu(
          kernel_agent, reinterpret_cast<void *>(server_address),
          reinterpret_cast<const void *>(vc), sizeof(p.instance.server));
      if (rc != 0)
        {
          fprintf(stderr, "Failed to copy server state to gpu\n");
          exit(1);
        }
    }

    // 4 bytes in its own page isn't ideal
    auto control_state =
        hsa::allocate(fine_grained_region, sizeof(HOSTRPC_ATOMIC(uint32_t)));

    kernel_args example = {
        .control = new (control_state.get()) HOSTRPC_ATOMIC(uint32_t),
        .application_args = 0,  // unused for now
    };

    const uint32_t init_control = 32;
    platform::atomic_store<uint32_t, __ATOMIC_RELEASE,
                           __OPENCL_MEMORY_SCOPE_ALL_SVM_DEVICES>(
        example.control, init_control);
    platform::fence_release();

    std::vector<launch_t<kernel_args>> l;
    // works on r > 1, can hit interesting behaviour for large r:
    // illegal instruction
    // HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION
    // memory access fault
    for (unsigned r = 0; r < 1; r++)
      {
        hsa_signal_t sig = {.handle = 0};

        l.emplace_back((launch_t<kernel_args>){
            kernel_agent, queue, std::move(sig),
            /* kernel_address */ kernel_address,
            kernel_private_segment_fixed_size, kernel_group_segment_fixed_size,
            /* number waves */ 1, example});
      }
    fprintf(stderr, "Launched instance\n");

    hostrpc::page_t tmp;
    memset(&tmp, 0, sizeof(tmp));

    for (unsigned i = 0; i < init_control; i++)
      {
        uint64_t expect[64] = {0};
        for (unsigned j = 0; j < 64; j++)
          {
            hostrpc::cacheline_t *t = &tmp.cacheline[j];
            t->element[0] = i;
            t->element[1] = i + 2;
            t->element[2] = i * 3;
            t->element[3] = i - 1;

            expect[j] =
                t->element[0] * t->element[1] * t->element[2] * t->element[3];
          }

        bool r = p.rpc_invoke<true>(&tmp, &tmp);

        for (unsigned j = 0; j < 64; j++)
          {
            if (tmp.cacheline[j].element[0] != expect[j])
              {
                fprintf(stderr, "Run[%u], fail at %u: %lu != %lu\n", i, j,
                        tmp.cacheline[j].element[0], expect[j]);
              }
          }

        fprintf(stderr, "rpc_invoke[%u] ret %u, tmp[0][0] %lu\n", i, r,
                tmp.cacheline[0].element[0]);
      }

    uint32_t ld = init_control + 1;
    while (ld != 0)
      {
        platform::fence_acquire();

        uint32_t nld =
            platform::atomic_load<uint32_t, __ATOMIC_ACQUIRE,
                                  __OPENCL_MEMORY_SCOPE_ALL_SVM_DEVICES>(
                example.control);

        if (nld != ld)
          {
            ld = nld;
            fprintf(stderr, "watching control, ld = %u\n", ld);
          }
      }

    fprintf(stderr, "Instance set control to non-zero\n");

    {
      auto c =
          hsa::allocate(fine_grained_region,
                        sizeof(hostrpc::gcn_x64_t::gcn_x64_type::server_type));
      void *vc = c.get();

      int rc =
          hsa::copy_host_to_gpu(kernel_agent, reinterpret_cast<void *>(vc),
                                reinterpret_cast<const void *>(server_address),

                                sizeof(p.instance.server));
      if (rc != 0)
        {
          fprintf(stderr, "Failed to copy server state back from gpu\n");
          exit(1);
        }

      memcpy(&p.instance.server, vc, sizeof(p.instance.server));
    }

    p.server_counters().dump();
    p.client_counters().dump();

    // hsa_queue_destroy(queue); // segv, probably means better counting needed
  }
}

#endif

#endif
