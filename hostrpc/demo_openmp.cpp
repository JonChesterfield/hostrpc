#pragma omp declare target

#include "allocator.hpp"
#include "base_types.hpp"
#include "client_server_pair.hpp"
#include "detail/client_impl.hpp"
#include "detail/server_impl.hpp"
#include "host_client.hpp"

namespace hostrpc
{
template <typename SZ, int device_num>
using x64_device_type_base =
    client_server_pair_t<SZ, copy_functor_given_alias, uint64_t,
                         allocator::openmp_shared<alignof(page_t)>,
                         allocator::openmp_shared<64>, allocator::host_libc<64>,
                         allocator::openmp_device<64, device_num>>;

template <typename SZ, int device_num>
struct x64_device_type : public x64_device_type_base<SZ, device_num>
{
  using base = x64_device_type_base<SZ, device_num>;
  HOSTRPC_ANNOTATE x64_device_type(SZ sz)
      : base(sz, typename base::AllocBuffer(),
             typename base::AllocInboxOutbox(), typename base::AllocLocal(),
             typename base::AllocRemote())
  {
  }
};
}  // namespace hostrpc

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

#include <omp.h>

#include "hostrpc_thread.hpp"
#include "openmp_plugins.hpp"
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

using SZ = hostrpc::size_compiletime<1920>;
constexpr static int device_num = 0;

using base_type = hostrpc::x64_device_type<SZ, device_num>;

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

  hostrpc::plugins got = hostrpc::find_plugins();

  fprintf(stderr, "amd: %u, ptx: %u\n", got.amdgcn, got.nvptx);

  {
    printf("in openmp host\n");
    SZ sz;

    base_type p(sz);
    p.storage.dump();

    auto serv_func =

        [&]() {
          fprintf(stderr, "thread lives\n");
          p.storage.dump();
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
        };

    auto serv = hostrpc::make_thread(&serv_func);

    client_instance = p.client;

    auto allocator =
        hostrpc::allocator::openmp_device<alignof(hostrpc::page_t), 0>();
    auto scratch_raw = allocator.allocate(sizeof(hostrpc::page_t));
    if (!scratch_raw.valid())
      {
        exit(1);
      }

    hostrpc::page_t *scratch =
        new (reinterpret_cast<hostrpc::page_t *>(scratch_raw.remote_ptr().ptr))
            hostrpc::page_t;

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

#if 1
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
#endif

    fprintf(stderr, "Post target region\n");

    serv.join();
    fprintf(stderr, "Joined\n");
    scratch_raw.destroy();
  }
}
