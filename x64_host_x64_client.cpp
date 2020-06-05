#include "catch.hpp"
#include "client.hpp"
#include "server.hpp"
#include "tests.hpp"

#include <cstring>

namespace
{
struct fill
{
  void operator()(hostrpc::page_t *page, void *dv)
  {
    __builtin_memcpy(page, dv, sizeof(hostrpc::page_t));
  };
};

struct use
{
  void operator()(hostrpc::page_t *page, void *dv)
  {
    __builtin_memcpy(dv, page, sizeof(hostrpc::page_t));
  };
};

struct operate
{
  void operator()(hostrpc::page_t *page, void *)
  {
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
  }
};

}  // namespace

using x64_x64_client = hostrpc::client<128, hostrpc::copy_functor_memcpy_pull,
                                       fill, use, hostrpc::nop_stepper>;

using x64_x64_server = hostrpc::server<128, hostrpc::copy_functor_memcpy_pull,
                                       operate, hostrpc::nop_stepper>;

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
slot_owner tracker;
}  // namespace hostrpc

TEST_CASE("hazard")
{
  using namespace hostrpc;
  mailbox_t<128> send;
  mailbox_t<128> recv;
  page_t client_buffer[128];
  page_t server_buffer[128];

  hostrpc::copy_functor_memcpy_pull cp;
  hostrpc::nop_stepper st;

  slot_bitmap<128, __OPENCL_MEMORY_SCOPE_DEVICE> client_active;
  slot_bitmap<128, __OPENCL_MEMORY_SCOPE_DEVICE> server_active;

  // auto send_data = mailbox_t<128>::slot_bitmap_data::alloc();

  x64_x64_client client(cp, recv, send, client_active, &server_buffer[0],
                        &client_buffer[0], st, fill{}, use{});

  x64_x64_server server(cp, send, recv, server_active, &client_buffer[0],
                        &server_buffer[0], st, operate{});

  _Atomic bool server_live(true);

  auto server_worker = [&](unsigned id) {
    my_id = id;
    unsigned count = 0;
    unsigned since_work = 0;
    for (;;)
      {
        if (!server_live)
          {
            printf("server %u did %u tasks\n", id, count);
            break;
          }
        bool did_work = server.rpc_handle(nullptr);
        if (did_work)
          {
            count++;
          }
        else
          {
            since_work++;
            platform::sleep_briefly();
          }

        if (since_work == 200000)
          {
            since_work = 0;

            if (id == 0)
              {
                printf("server %u stalled\n", id);
                printf("i:   ");
                send.dump();
                printf("o:   ");
                recv.dump();
                printf("a: ");
                server_active.dump();
                tracker.dump();
              }
          }
      }
  };

  unsigned nservers = 64;
  unsigned nclients = 64;

  auto client_worker = [&](unsigned id, unsigned reps) {
    my_id = id;
    page_t scratch;
    page_t expect;
    unsigned count = 0;
    unsigned since_work = 0;
    for (unsigned r = 0; r < reps; r++)
      {
        init_page(&scratch, id);
        init_page(&expect, id + 1);
        if (client.rpc_invoke<true>(&scratch))
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
        else
          {
            since_work++;
            platform::sleep_briefly();
          }

        if (since_work == 10000)
          {
            since_work = 0;
            printf("client %u stalled\n", id);
          }
      }

    printf("client %u ran %u / %u reps\n", id, count, reps);
  };

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
