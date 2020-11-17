#pragma omp declare target
#include "detail/client_impl.hpp"
#include "detail/server_impl.hpp"

#include "allocator.hpp"
#include "host_client.hpp"

#include "x64_gcn_type.hpp"
#include "x64_target_type.hpp"

#include <cinttypes>
#include <cstdlib>

static void copy_page(hostrpc::page_t *dst, hostrpc::page_t *src)
{
  unsigned id = platform::get_lane_id();
  hostrpc::cacheline_t *dline = &dst->cacheline[id];
  hostrpc::cacheline_t *sline = &src->cacheline[id];
  for (unsigned e = 0; e < 8; e++)
    {
      dline->element[e] = sline->element[e];
    }
}

struct fill
{
  fill(hostrpc::page_t *d) : d(d) {}
  hostrpc::page_t *d;

  void operator()(hostrpc::page_t *page) { copy_page(page, d); };
};

struct use
{
  use(hostrpc::page_t *d) : d(d) {}
  hostrpc::page_t *d;

  void operator()(hostrpc::page_t *page) { copy_page(d, page); };
};

#pragma omp end declare target

// this fails to compile - no member named 'printf' in the global namespace
// seems to be trying to use stuff from wchar, can probably work around by
// using pthreads instead (as thread includes string which seems to be the
// problem)

#include <stdio.h>
#include <thread>
#include <unistd.h>

#define WITH_HSA 1

using SZ = hostrpc::size_compiletime<1920>;

#if WITH_HSA
using base_type = hostrpc::x64_gcn_type<SZ>;
#include "hsa.hpp"
#else
using base_type = hostrpc::x64_target_type<0>;
#endif

base_type::client_type client_instance;

struct operate_test
{
  void operator()(hostrpc::page_t *) { fprintf(stderr, "Invoked operate\n"); }
};
struct clear_test
{
  void operator()(hostrpc::page_t *) { fprintf(stderr, "Invoked clear\n"); }
};

int main()
{
#pragma omp target
  asm("// less lazy");

#if WITH_HSA
  hsa::init hsa;
#endif
  {
    printf("in openmp host\n");
    SZ sz;

#if WITH_HSA
    hsa_agent_t kernel_agent = hsa::find_a_gpu_or_exit();
    hsa_region_t fine_grained_region = hsa::region_fine_grained(kernel_agent);
    hsa_region_t coarse_grained_region =
        hsa::region_coarse_grained(kernel_agent);
    base_type p(sz, fine_grained_region.handle, coarse_grained_region.handle);
#else
    fprintf(stderr, "base type before\n");
    base_type p(sz);
    fprintf(stderr, "base type after\n");
#endif
    std::thread serv([&]() {
      uint32_t location = 0;

      for (unsigned i = 0; i < 16; i++)
        {
          bool r = p.server.rpc_handle<operate_test, clear_test>(
              operate_test{}, clear_test{}, &location);
          fprintf(stderr, "server ret %u\n", r);
          for (unsigned j = 0; j < 1000; j++)
            {
              platform::sleep();
            }
        }
    });

    client_instance = p.client;

    auto allocator =
        hostrpc::allocator::openmp_target<alignof(hostrpc::page_t), 0>();
    auto scratch_raw = allocator.allocate(sizeof(hostrpc::page_t));
    if (!scratch_raw.valid())
      {
        exit(1);
      }

    hostrpc::page_t *scratch =
        new (scratch_raw.remote_ptr().ptr) hostrpc::page_t;

    fprintf(stderr, "scratch %p\n", scratch);

    printf("remote_buffer 0x%.12" PRIxPTR "\n",
           (uintptr_t)client_instance.remote_buffer);
    printf("local_buffer  0x%.12" PRIxPTR "\n",
           (uintptr_t)client_instance.local_buffer);
    printf("inbox         0x%.12" PRIxPTR "\n",
           (uintptr_t)client_instance.inbox.a);
    printf("outbox        0x%.12" PRIxPTR "\n",
           (uintptr_t)client_instance.outbox.a);
    printf("active        0x%.12" PRIxPTR "\n",
           (uintptr_t)client_instance.active.a);
    printf("outbox stg    0x%.12" PRIxPTR "\n",
           (uintptr_t)client_instance.staging.a);

#pragma omp target map(tofrom \
                       : client_instance) device(0) is_device_ptr(scratch)
    {
      printf("gpu: scratch %p\n", scratch);

      printf("remote_buffer 0x%.12" PRIxPTR "\n",
             (uintptr_t)client_instance.remote_buffer);
      printf("local_buffer  0x%.12" PRIxPTR "\n",
             (uintptr_t)client_instance.local_buffer);
      printf("inbox         0x%.12" PRIxPTR "\n",
             (uintptr_t)client_instance.inbox.a);
      printf("outbox        0x%.12" PRIxPTR "\n",
             (uintptr_t)client_instance.outbox.a);
      printf("active        0x%.12" PRIxPTR "\n",
             (uintptr_t)client_instance.active.a);
      printf("outbox stg    0x%.12" PRIxPTR "\n",
             (uintptr_t)client_instance.staging.a);

      fill f(scratch);
      use u(scratch);
      client_instance.rpc_invoke<fill, use, true>(f, u);
    }

    serv.join();

    scratch_raw.destroy();
  }
}
