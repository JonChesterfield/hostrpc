#include "catch.hpp"
#include "client.hpp"
#include "server.hpp"
#include "tests.hpp"

#include <cstring>

namespace
{
struct fill
{
static void call(hostrpc::page_t *page, void *dv)
  {
    __builtin_memcpy(page, dv, sizeof(hostrpc::page_t));
  };
};

struct use
{
 static void call (hostrpc::page_t *page, void *dv)
  {
    __builtin_memcpy(dv, page, sizeof(hostrpc::page_t));
  };
};

struct operate
{
  static void call(hostrpc::page_t *page, void *)
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

using x64_x64_client =
    hostrpc::client<128, hostrpc::x64_x64_bitmap_types,
                    hostrpc::copy_functor_memcpy_pull, fill,
                    use, hostrpc::nop_stepper>;

using x64_x64_server =
    hostrpc::server<128, hostrpc::x64_x64_bitmap_types,
                    hostrpc::copy_functor_memcpy_pull, operate,
                    hostrpc::nop_stepper>;

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
  constexpr size_t N = 128;
  page_t client_buffer[N];
  page_t server_buffer[N];

  using mailbox_ptr_t =
      std::unique_ptr<mailbox_t<N>::slot_bitmap_data_t,
                      mailbox_t<N>::slot_bitmap_data_t::deleter>;

  using lockarray_ptr_t =
      std::unique_ptr<lockarray_t<N>::slot_bitmap_data_t,
                      lockarray_t<N>::slot_bitmap_data_t::deleter>;

  mailbox_ptr_t send_data(mailbox_t<N>::slot_bitmap_data_t::alloc());
  mailbox_ptr_t recv_data(mailbox_t<N>::slot_bitmap_data_t::alloc());
  lockarray_ptr_t client_active_data(
      lockarray_t<N>::slot_bitmap_data_t::alloc());
  lockarray_ptr_t server_active_data(
      lockarray_t<N>::slot_bitmap_data_t::alloc());

  mailbox_t<N> send(send_data.get());
  mailbox_t<N> recv(recv_data.get());
  lockarray_t<N> client_active(client_active_data.get());
  lockarray_t<N> server_active(server_active_data.get());

  x64_x64_client client(recv, send, client_active, &server_buffer[0],
                        &client_buffer[0]);

  x64_x64_server server(send, recv, server_active, &client_buffer[0],
                        &server_buffer[0]);

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
