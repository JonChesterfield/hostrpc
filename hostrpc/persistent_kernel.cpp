#define VISIBLE __attribute__((visibility("default")))

#include "cxa_atexit.hpp"
#include "detail/platform_detect.hpp"

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

#include "detail/client_impl.hpp"
#include "detail/server_impl.hpp"

#include "allocator.hpp"
#include "host_client.hpp"

#include "memory.hpp"

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

#if defined(__AMDGCN__)
static void gcn_server_callback(hostrpc::cacheline_t *line)
{
  // not yet implemented, maybe take a function pointer out of [0]
  uint64_t l01 = line->element[0] * line->element[1];
  uint64_t l23 = line->element[2] * line->element[3];
  line->element[0] = l01 * l23;
}
#endif

struct gcn_x64_type
{
  using SZ = hostrpc::size_runtime;
  using Word = uint64_t;

  using client_type = client_impl<Word, SZ>;
  using server_type = server_impl<Word, SZ>;

  client_type client;
  server_type server;

  using AllocBuffer = hostrpc::allocator::hsa<alignof(page_t)>;
  using AllocInboxOutbox = hostrpc::allocator::hsa<64>;

  using AllocLocal = hostrpc::allocator::hsa<64>;
  using AllocRemote = hostrpc::allocator::host_libc<64>;

  using storage_type = allocator::store_impl<AllocBuffer, AllocInboxOutbox,
                                             AllocLocal, AllocRemote>;

  storage_type storage;

  gcn_x64_type(SZ sz, uint64_t fine_handle, uint64_t coarse_handle)
      : storage(host_client(
            AllocBuffer(fine_handle), AllocInboxOutbox(fine_handle),
            AllocLocal(coarse_handle), AllocRemote(), sz, &server, &client))
  {
  }

  ~gcn_x64_type() { storage.destroy(); }
};

}  // namespace hostrpc

#if !defined(__AMDGCN__)
#include "amd_hsa_queue.h"
#include "hsa.h"
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
VISIBLE hostrpc::gcn_x64_type::server_type server_instance[1];

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
  struct operate
  {
    void operator()(hostrpc::page_t *page)
    {
      // Call through to a specific handler, one cache line per lane
      hostrpc::cacheline_t *l = &page->cacheline[platform::get_lane_id()];
      gcn_server_callback(l);
    };
  } op;

  struct clear
  {
    void operator()(hostrpc::page_t *page)
    {
      hostrpc::cacheline_t *l = &page->cacheline[platform::get_lane_id()];
      for (unsigned i = 0; i < 8; i++)
        {
          l->element[i] = 0;
        }
    }
  } cl;

  if (server_instance[0].rpc_handle(op, cl, &location_arg))
    {
      // did work
    }
  else
    {
      // May have been no work to do because we're shutting down
      uint64_t ctrl =
          platform::atomic_load<uint32_t, __ATOMIC_RELAXED,
                                __OPENCL_MEMORY_SCOPE_ALL_SVM_DEVICES>(control);
      if (ctrl == 0)
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

#if HOSTRPC_HOST
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

    hsa_queue_t *queue = hsa::create_queue(kernel_agent);
    if (!queue)
    {
      fprintf(stderr, "Failed to create queue\n");
      exit(1);
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
    hostrpc::gcn_x64_type p(hostrpc::round64(N), fine_grained_region.handle,
                            coarse_grained_region.handle);

    {
      auto c = hsa::allocate(fine_grained_region,
                             sizeof(hostrpc::gcn_x64_type::server_type));
      void *vc = c.get();
      memcpy(vc, &p.server, sizeof(p.server));

      int rc = hsa::copy_host_to_gpu(
          kernel_agent, reinterpret_cast<void *>(server_address),
          reinterpret_cast<const void *>(vc), sizeof(p.server));
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
        example.control, 1);
    platform::fence_release();

    std::vector<launch_t<kernel_args>> l;
    // works on r > 1, can hit interesting behaviour for large r:
    // illegal instruction
    // HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION
    // memory access fault
    for (unsigned r = 0; r < 4; r++)
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

    struct fill
    {
      hostrpc::page_t *d;
      fill(hostrpc::page_t *d) : d(d) {}
      void operator()(hostrpc::page_t *page)
      {
        __builtin_memcpy(page, d, sizeof(hostrpc::page_t));
      };
    } f(&tmp);

    struct use
    {
      hostrpc::page_t *d;
      use(hostrpc::page_t *d) : d(d) {}
      void operator()(hostrpc::page_t *page)
      {
        __builtin_memcpy(d, page, sizeof(hostrpc::page_t));
      };
    } u(&tmp);

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

        bool r = p.client.rpc_invoke<decltype(f), decltype(u)>(f, u);

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

    // Client calls finished, tell server to wind down
    platform::atomic_store<uint32_t, __ATOMIC_RELEASE,
                           __OPENCL_MEMORY_SCOPE_ALL_SVM_DEVICES>(
        example.control, 0);

    fprintf(stderr, "Server told to wind down\n");

    {
      auto c = hsa::allocate(fine_grained_region,
                             sizeof(hostrpc::gcn_x64_type::server_type));
      void *vc = c.get();

      int rc =
          hsa::copy_host_to_gpu(kernel_agent, reinterpret_cast<void *>(vc),
                                reinterpret_cast<const void *>(server_address),

                                sizeof(p.server));
      if (rc != 0)
        {
          fprintf(stderr, "Failed to copy server state back from gpu\n");
          exit(1);
        }

      memcpy(&p.server, vc, sizeof(p.server));
    }

    p.server.get_counters().dump();
    p.client.get_counters().dump();

    // the wait() on the launch_t does nothing because there is no completion
    // signal. This is therefore racy - need the server instance to report that
    // it knows it is shutting down and will not read any memory, or possibly
    // further - might not be able to tear down the queue before we know the
    // kernel has finished
    sleep(1);

    l.clear();
    hsa_queue_destroy(
        queue);  // segv reasing packet->completion_signal in ~launch_t
  }
}

#endif

#endif
