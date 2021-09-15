#include "base_types.hpp"
#include "catch.hpp"
#include "detail/platform_detect.hpp"
#include "pool_interface.hpp"

#include <array>
#include <cstring>
#include <thread>

#include "allocator.hpp"
#include "client_server_pair.hpp"
#include "detail/client_impl.hpp"
#include "detail/server_impl.hpp"
#include "host_client.hpp"

static void init_page(hostrpc::page_t *page, uint64_t v)
{
  for (unsigned i = 0; i < 64; i++)
    {
      for (unsigned e = 0; e < 8; e++)
        {
          page->cacheline[i].element[e] = v;
        }
    }
}

enum
{
  maximum_threads = 1024
};

POOL_INTERFACE_BOILERPLATE_HOST(stress_pool_server, maximum_threads);
POOL_INTERFACE_BOILERPLATE_HOST(stress_pool_client, maximum_threads);

HOSTRPC_ATOMIC(uint64_t) *client_to_run = nullptr;
HOSTRPC_ATOMIC(uint64_t) *server_ran = nullptr;

namespace hostrpc
{
using x64_x64_type_base =
    client_server_pair_t<hostrpc::size_runtime<uint32_t>, uint64_t,
                         hostrpc::allocator::host_libc<alignof(page_t)>,
                         hostrpc::allocator::host_libc<64>,
                         hostrpc::allocator::host_libc<64>,
                         hostrpc::allocator::host_libc<64> >;

struct x64_x64_type : public x64_x64_type_base
{
  using base = x64_x64_type_base;
  HOSTRPC_ANNOTATE x64_x64_type(size_t N)
      : base(hostrpc::size_runtime<uint32_t>(N), typename base::AllocBuffer(),
             typename base::AllocInboxOutbox(), typename base::AllocLocal(),
             typename base::AllocRemote())
  {
  }
};

}  // namespace hostrpc

using type_under_test = hostrpc::x64_x64_type;

type_under_test p(100);

uint32_t stress_pool_server::run(uint32_t server_location)
{
  auto op_func = [](uint32_t, hostrpc::page_t *page) {
    for (unsigned c = 0; c < 64; c++)
      {
        hostrpc::cacheline_t &line = page->cacheline[c];
        for (unsigned i = 0; i < 8; i++)
          {
            line.element[i]++;
          }
      }
  };

  auto cl_func = [](uint32_t, hostrpc::page_t *page) {
    for (unsigned c = 0; c < 64; c++)
      {
        hostrpc::cacheline_t &line = page->cacheline[c];
        for (unsigned i = 0; i < 8; i++)
          {
            line.element[i] = 0;
          }
      }
  };

  bool did_work = p.server.rpc_handle(op_func, cl_func, &server_location);
  (void)did_work;

  if (did_work)
    {
      if (0)
        printf("server %u at loc %u progressed\n", get_current_uuid(),
               server_location);
    }

  return server_location;
}

uint32_t stress_pool_client::run(uint32_t state)
{
  uint32_t id = get_current_uuid();

  hostrpc::page_t scratch;
  hostrpc::page_t expect;

  if (client_to_run[id] == 0)
    {
      return state;
    }
  else
    {
      platform::atomic_fetch_sub<uint64_t, __ATOMIC_ACQ_REL,
                                 __OPENCL_MEMORY_SCOPE_ALL_SVM_DEVICES>(
          &client_to_run[id], 1);
    }

  auto fill = [&](uint32_t, hostrpc::page_t *page) {
    __builtin_memcpy(page, &scratch, sizeof(hostrpc::page_t));
  };
  auto use = [&](uint32_t, hostrpc::page_t *page) {
    __builtin_memcpy(&scratch, page, sizeof(hostrpc::page_t));
  };

  init_page(&scratch, id);
  init_page(&expect, id + 1);

  if (p.client.rpc_invoke(fill, use))
    {
      if (__builtin_memcmp(&scratch, &expect, sizeof(hostrpc::page_t)) != 0)
        {
          printf("client %u error: ", id);
          printf("%lu vs %lu\n", scratch.cacheline[0].element[0],
                 expect.cacheline[0].element[0]);
          state++;
        }
    }

  return state;
}

TEST_CASE("x64_x64_stress")
{
  using namespace hostrpc;

  enum
  {
    client_count = 128 * 8,
    server_count = 128 * 8,
    reps_per_client = 8192,
  };

  auto alloc = type_under_test::base::AllocBuffer();

  auto raw_client_to_run =
      alloc.allocate(sizeof(HOSTRPC_ATOMIC(uint64_t)) * maximum_threads);
  auto raw_server_ran =
      alloc.allocate(sizeof(HOSTRPC_ATOMIC(uint64_t)) * maximum_threads);

  if (!raw_client_to_run.valid() || !raw_server_ran.valid())
    {
      printf("Memory allocation failure, aborting\n");
      return;
    }

  client_to_run = reinterpret_cast<HOSTRPC_ATOMIC(uint64_t) *>(
      static_cast<void *>(raw_client_to_run.local_ptr()));
  server_ran = reinterpret_cast<HOSTRPC_ATOMIC(uint64_t) *>(
      static_cast<void *>(raw_server_ran.local_ptr()));

  for (uint64_t i = 0; i < maximum_threads; i++)
    {
      client_to_run[i] = (i < client_count) ? reps_per_client : 0;
      server_ran[i] = 0;
    }

  auto remainder = [&](const char *loc, uint64_t counter = 0) -> bool {
    platform::fence_release();
    uint64_t remaining = 0;
    for (uint64_t i = 0; i < maximum_threads; i++)
      {
        remaining += client_to_run[i];
      }
    printf("Remaining (%s/%lu): %lu\n", loc, counter, remaining);
    platform::fence_acquire();
    return remaining;
  };

  remainder("Init");

  stress_pool_server::bootstrap_entry(server_count);
  stress_pool_client::bootstrap_entry(client_count);

  for (uint64_t c = 0;; c++)
    {
      uint64_t r = remainder("Running", c);
      usleep(100000);
      if (r == 0)
        {
          break;
        }
    }

  stress_pool_client::teardown();

  remainder("Client down");

  stress_pool_server::teardown();

  remainder("Server down");

  raw_client_to_run.destroy();
  raw_server_ran.destroy();
}
