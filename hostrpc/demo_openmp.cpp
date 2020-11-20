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

template <typename C, bool have_continuation>
static bool invoke(C *client, uint64_t x0, uint64_t x1, uint64_t x2,
                   uint64_t x3, uint64_t x4, uint64_t x5, uint64_t x6,
                   uint64_t x7)
{
  auto fill = [&](hostrpc::page_t *page) -> void {
    hostrpc::cacheline_t *line = &page->cacheline[platform::get_lane_id()];
    line->element[0] = x0;
    line->element[1] = x1;
    line->element[2] = x2;
    line->element[3] = x3;
    line->element[4] = x4;
    line->element[5] = x5;
    line->element[6] = x6;
    line->element[7] = x7;
  };

  auto use = [&](hostrpc::page_t *page) -> void {
    hostrpc::cacheline_t *line = &page->cacheline[platform::get_lane_id()];
    x0 = line->element[0];
    x1 = line->element[1];
    x2 = line->element[2];
    x3 = line->element[3];
    x4 = line->element[4];
    x5 = line->element[5];
    x6 = line->element[6];
    x7 = line->element[7];
  };

  return client
      ->template rpc_invoke<decltype(fill), decltype(use), have_continuation>(
          fill, use);
}

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
  void operator()(hostrpc::page_t *page)
  {
    fprintf(stderr, "Invoked operate\n");
    for (unsigned i = 0; i < 64; i++)
      {
        operator()(i, &page->cacheline[i]);
      }
  }

  void operator()(unsigned index, hostrpc::cacheline_t *line)
  {
    printf("%u: (%lu, %lu, %lu, %lu, %lu, %lu, %lu, %lu)\n", index,
           line->element[0], line->element[1], line->element[2],
           line->element[3], line->element[4], line->element[5],
           line->element[6], line->element[7]);
  }
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

  fprintf(stderr, "amd: %u, ptx: %u. Found %u devices\n", got.amdgcn, got.nvptx,
          omp_get_num_devices());

  {
    SZ sz;

    base_type p(sz);

    HOSTRPC_ATOMIC(uint32_t) server_control;
    platform::atomic_store<uint32_t, __ATOMIC_RELEASE,
                           __OPENCL_MEMORY_SCOPE_ALL_SVM_DEVICES>(
        &server_control, 1);

    auto serv_func = [&]() {
      uint32_t location = 0;

      auto serv_func_busy = [&]() {
        bool r = true;
        while (r)
          {
            r = p.server.rpc_handle<operate_test, clear_test>(
                operate_test{}, clear_test{}, &location);
          }
      };

      for (;;)
        {
          serv_func_busy();

          // ran out of work, has client set control to cease?
          uint32_t ctrl =
              platform::atomic_load<uint32_t, __ATOMIC_ACQUIRE,
                                    __OPENCL_MEMORY_SCOPE_ALL_SVM_DEVICES>(
                  &server_control);

          if (ctrl == 0)
            {
              // client called to cease, empty any clear jobs in the pipeline
              serv_func_busy();
              break;
            }

          // nothing to do, but not told to stop. spin.
          for (unsigned j = 0; j < 1000; j++)
            {
              platform::sleep();
            }
        }
    };

    auto serv = hostrpc::make_thread(&serv_func);

    client_instance = p.client;

#pragma omp target parallel for map(tofrom : client_instance) device(0)
    for (int i = 0; i < 128; i++)
      {
        unsigned id = platform::get_lane_id();
        invoke<decltype(client_instance), true>(&client_instance, id, 6, 5, 4,
                                                3, 2, 1, 0);
      }

    fprintf(stderr, "Post target region\n");

    platform::atomic_store<uint32_t, __ATOMIC_RELEASE,
                           __OPENCL_MEMORY_SCOPE_ALL_SVM_DEVICES>(
        &server_control, 0);

    serv.join();
    fprintf(stderr, "Joined\n");
  }
}
