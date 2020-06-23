#include "base_types.hpp"
#include "catch.hpp"
#include "interface.hpp"

#include <cstring>
#include <thread>

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

namespace hostrpc
{
thread_local unsigned my_id = 0;
}  // namespace hostrpc

TEST_CASE("x64_x64_stress")
{
  using namespace hostrpc;
  hostrpc::x64_x64_t p(100);

  _Atomic bool server_live(true);

  auto op_func = [](hostrpc::page_t *page) {
    for (unsigned c = 0; c < 64; c++)
      {
        hostrpc::cacheline_t &line = page->cacheline[c];
        std::swap(line.element[0], line.element[7]);
        std::swap(line.element[1], line.element[6]);
        std::swap(line.element[2], line.element[5]);
        std::swap(line.element[3], line.element[4]);
        for (unsigned i = 0; i < 8; i++)
          {
            line.element[i]++;
          }
      }
  };

  auto server_worker = [&](unsigned id) {
    my_id = id;
    unsigned count = 0;
    unsigned since_work = 0;

    uint64_t server_location = 0;
    for (;;)
      {
        if (!server_live)
          {
            printf("server %u did %u tasks\n", id, count);
            break;
          }
        bool did_work = p.server().handle(op_func, &server_location);
        if (did_work)
          {
            count++;
          }
        else
          {
            since_work++;
          }
      }
  };

  auto client_worker = [&](unsigned id, unsigned reps) {
    my_id = id;
    page_t scratch;
    page_t expect;
    unsigned count = 0;
    for (unsigned r = 0; r < reps; r++)
      {
        init_page(&scratch, id);
        init_page(&expect, id + 1);
        if (p.client().invoke(
                [&](hostrpc::page_t *page) {
                  __builtin_memcpy(page, &scratch, sizeof(hostrpc::page_t));
                },
                [&](hostrpc::page_t *page) {
                  __builtin_memcpy(&scratch, page, sizeof(hostrpc::page_t));
                }))
          {
            count++;
            if (memcmp(&scratch, &expect, sizeof(page_t)) != 0)
              {
                printf("client %u error: ", id);
                printf("%lu vs %lu\n", scratch.cacheline[0].element[0],
                       expect.cacheline[0].element[0]);

                return;
              }
          }
      }

    printf("client %u ran %u / %u reps\n", id, count, reps);
  };

  unsigned nservers = 64;
  unsigned nclients = 64;

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
}
