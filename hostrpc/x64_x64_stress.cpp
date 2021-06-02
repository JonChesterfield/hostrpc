#include "base_types.hpp"
#include "catch.hpp"
#include "detail/platform_detect.hpp"
#include "pool_interface.hpp"

#include <cstring>
#include <thread>

#include "allocator.hpp"
#include "client_server_pair.hpp"
#include "detail/client_impl.hpp"
#include "detail/server_impl.hpp"
#include "host_client.hpp"

namespace hostrpc
{
using x64_x64_type_base =
    client_server_pair_t<hostrpc::size_runtime, uint64_t,
                         hostrpc::allocator::host_libc<alignof(page_t)>,
                         hostrpc::allocator::host_libc<64>,
                         hostrpc::allocator::host_libc<64>,
                         hostrpc::allocator::host_libc<64> >;

struct x64_x64_type : public x64_x64_type_base
{
  using base = x64_x64_type_base;
  HOSTRPC_ANNOTATE x64_x64_type(size_t N)
      : x64_x64_type_base(
            hostrpc::size_runtime(N), typename base::AllocBuffer(),
            typename base::AllocInboxOutbox(), typename base::AllocLocal(),
            typename base::AllocRemote())
  {
  }
};

}  // namespace hostrpc

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

POOL_INTERFACE_BOILERPLATE_HOST(stress_pool, 1024);

TEST_CASE("x64_x64_stress")
{
  using namespace hostrpc;

  hostrpc::x64_x64_type p(100);

  auto op_func = [](hostrpc::page_t *page) {
    for (unsigned c = 0; c < 64; c++)
      {
        hostrpc::cacheline_t &line = page->cacheline[c];
        for (unsigned i = 0; i < 8; i++)
          {
            line.element[i]++;
          }
      }
  };

  auto cl_func = [](hostrpc::page_t *page) {
    for (unsigned c = 0; c < 64; c++)
      {
        hostrpc::cacheline_t &line = page->cacheline[c];
        for (unsigned i = 0; i < 8; i++)
          {
            line.element[i] = 0;
          }
      }
  };

  HOSTRPC_ATOMIC(bool) server_live(true);

  auto server_worker = [&](unsigned id) {
    unsigned count = 0;

    uint32_t server_location = 0;
    for (;;)
      {
        if (!server_live)
          {
            printf("server %u did %u tasks\n", id, count);
            break;
          }
        bool did_work =
            p.server.rpc_handle<decltype(op_func), decltype(cl_func)>(
                op_func, cl_func, &server_location);
        if (did_work)
          {
            count++;
          }
      }
  };

  // makes a copy, which is cheap but not free
  // when using counters, the copy means all the clients use their
  // own counter - which is good for efficiency but complicates reporting

  auto client_worker = [&](unsigned id, unsigned reps) -> unsigned {
    page_t scratch;
    page_t expect;
    unsigned count = 0;
    unsigned failures = 0;

    auto fill = [&](hostrpc::page_t *page) {
      __builtin_memcpy(page, &scratch, sizeof(hostrpc::page_t));
    };
    auto use = [&](hostrpc::page_t *page) {
      __builtin_memcpy(&scratch, page, sizeof(hostrpc::page_t));
    };

    for (unsigned r = 0; r < reps; r++)
      {
        init_page(&scratch, id + r);
        init_page(&expect, id + r + 1);

        if (p.client.rpc_invoke<decltype(fill), decltype(use)>(fill, use))
          {
            count++;
            if (__builtin_memcmp(&scratch, &expect, sizeof(hostrpc::page_t)) !=
                0)
              {
                failures++;
                printf("client %u error: ", id);
                printf("%lu vs %lu\n", scratch.cacheline[0].element[0],
                       expect.cacheline[0].element[0]);
                return failures;
              }
          }
      }

    printf("client %u ran %u / %u reps with %u failures\n", id, count, reps,
           failures);
    return failures;
  };

  unsigned nservers = 32;
  unsigned nclients = 32;  // was 128

  std::vector<std::thread> server_store;
  for (unsigned i = 0; i < nservers; i++)
    {
      server_store.emplace_back(std::thread(server_worker, i));
    }

  std::vector<std::thread> client_store;
  for (unsigned i = 0; i < nclients; i++)
    {
      client_store.emplace_back(
          std::thread(client_worker, i + nservers, 10000));
    }

  for (auto &i : client_store)
    {
      i.join();
    }

  printf("client's joined\n");
  server_live = false;
  for (auto &i : server_store)
    {
      i.join();
    }

  printf("x64_x64_stress counters:\n");
  p.client_counters().dump();
}
