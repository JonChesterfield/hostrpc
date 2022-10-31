#define HOSTRPC_HAVE_STDIO 1
#include <stdio.h>

#include <omp.h>


#pragma omp declare target

#include "allocator.hpp"
#include "base_types.hpp"
#include "client_server_pair.hpp"

#if HOSTRPC_HOST
// Easier than linking host bitcode for x64
#include "allocator_host_libc.cpp"
#endif

namespace hostrpc
{
template <typename SZ, int device_num>
using x64_device_type_base =
    client_server_pair_t<SZ, arch::x64, arch::openmp_target<device_num> >;

template <typename SZ, int device_num>
struct x64_device_type : public x64_device_type_base<SZ, device_num>
{
  using base = x64_device_type_base<SZ, device_num>;
  HOSTRPC_ANNOTATE x64_device_type(SZ sz) : base(sz, {}, {}) {}
};
}  // namespace hostrpc

template <typename C>
static bool invoke(C *client, uint64_t x[8])
{
  auto fill = [&](hostrpc::port_t, hostrpc::page_t *page) -> void {
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

  auto use = [&](hostrpc::port_t, hostrpc::page_t *page) -> void {
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

  return client->template rpc_invoke(fill, use);
}

#pragma omp end declare target

#include <omp.h>

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <x86_64-linux-gnu/asm/unistd_64.h>

using SZ = hostrpc::size_compiletime<1920>;
constexpr static int device_num = 0;

using base_type = hostrpc::x64_device_type<SZ, device_num>;

struct operate_test
{
  void operator()(hostrpc::port_t port, hostrpc::page_t *page)
  {
    fprintf(stderr, "Invoked operate\n");
    for (unsigned i = 0; i < 64; i++)
      {
        operator()(port, i, &page->cacheline[i]);
      }
  }

  void operator()(hostrpc::port_t, unsigned index, hostrpc::cacheline_t *line)
  {
#if HOSTRPC_HOST
    // hostrpc::syscall_on_cache_line(index, line);
#endif
  }
};

struct clear_test
{
  void operator()(hostrpc::port_t, hostrpc::page_t *page)
  {
    for (unsigned c = 0; c < 64; c++)
      {
        hostrpc::cacheline_t &line = page->cacheline[c];
        line.element[0] = 0;
      }
  }
};

base_type::server_type global_server;

#pragma omp declare target
base_type::client_type global_client;
#pragma omp end declare target

int main()
{
  SZ sz;
  base_type p(sz);

  if (!p.valid())
    {
      fprintf(stderr, "%s: Failed to allocate\n", __func__);
      return 1;
    }
  
  global_server = p.server;
  global_client = p.client;
    
#pragma omp target data map(to: global_client)
    {}

#pragma omp parallel num_threads(4)
    {
      printf("on the host\n");

#if 0
#pragma omp target device(0)
    {
      auto inv = [&](uint64_t x[8]) -> bool {
        return invoke<base_type::client_type>(global_client, x);
      };

    inv({1,2,3});
    }
#endif
    }
    
}
