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
static bool invoke(C *client, uint64_t x[8])
{
  auto fill = [&](hostrpc::page_t *page) -> void {
    hostrpc::cacheline_t *line = &page->cacheline[platform::get_lane_id()];
    line->element[0] = x[0];
    line->element[1] = x[1];
    line->element[2] = x[2];
    line->element[3] = x[3];
    line->element[4] = x[4];
    line->element[5] = x[5];
    line->element[6] = x[6];
    line->element[7] = x[7];
  };

  auto use = [&](hostrpc::page_t *page) -> void {
    hostrpc::cacheline_t *line = &page->cacheline[platform::get_lane_id()];
    x[0] = line->element[0];
    x[1] = line->element[1];
    x[2] = line->element[2];
    x[3] = line->element[3];
    x[4] = line->element[4];
    x[5] = line->element[5];
    x[6] = line->element[6];
    x[7] = line->element[7];
  };

  return client
      ->template rpc_invoke<decltype(fill), decltype(use), have_continuation>(
          fill, use);
}

#pragma omp end declare target

#include <omp.h>

#include "hostrpc_thread.hpp"
#include "openmp_plugins.hpp"
#include "syscall.hpp"

#include "server_thread_state.hpp"

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <x86_64-linux-gnu/asm/unistd_64.h>

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
#if HOSTRPC_HOST
    hostrpc::syscall_on_cache_line(index, line);
#endif
  }
};

struct clear_test
{
  void operator()(hostrpc::page_t *page)
  {
    for (unsigned c = 0; c < 64; c++)
      {
        hostrpc::cacheline_t &line = page->cacheline[c];
        line.element[0] = hostrpc::no_op;
      }
  }
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

    if (!p.valid())
      {
        fprintf(stderr, "%s: Failed to allocate\n", __func__);
        return 1;
      }

    HOSTRPC_ATOMIC(uint32_t) server_control;
    platform::atomic_store<uint32_t, __ATOMIC_RELEASE,
                           __OPENCL_MEMORY_SCOPE_ALL_SVM_DEVICES>(
        &server_control, 1);

    auto s = hostrpc::make_server_thread_state(&p.server, &server_control,
                                               operate_test{}, clear_test{});
    auto serv = hostrpc::make_thread(&s);

    client_instance = p.client;

#pragma omp target map(tofrom : client_instance) device(0)
    {
      auto inv = [&](uint64_t x[8]) -> bool {
        return invoke<base_type::client_type, true>(&client_instance, x);
      };

      uint64_t tmp[8];
      tmp[0] = hostrpc::allocate_op;
      tmp[1] = 16;
      inv(tmp);

      char *buf = (char *)tmp[0];

      buf[0] = 'h';
      buf[1] = 'i';
      buf[2] = '\n';
      buf[3] = '\0';

      tmp[0] = hostrpc::syscall_op;
      tmp[1] = __NR_write;
      tmp[2] = 2;
      tmp[3] = (uint64_t)buf;
      tmp[4] = 3;

      inv(tmp);

      tmp[0] = hostrpc::syscall_op;
      tmp[1] = __NR_fsync;
      tmp[2] = 2;

      inv(tmp);

      tmp[0] = hostrpc::free_op;
      tmp[1] = (uint64_t)buf;
      inv(tmp);
    }

    fprintf(stderr, "Post target region\n");

    platform::atomic_store<uint32_t, __ATOMIC_RELEASE,
                           __OPENCL_MEMORY_SCOPE_ALL_SVM_DEVICES>(
        &server_control, 0);

    serv.join();
    fprintf(stderr, "Joined\n");
  }
}
