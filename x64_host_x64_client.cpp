#include "x64_host_x64_client.hpp"

#include "client.hpp"
#include "client_impl.hpp"
#include "memory.hpp"
#include "server_impl.hpp"

#include <string.h>

namespace hostrpc
{
x64_x64_t::client_t::client_t(size_t N) {}
x64_x64_t::client_t::client_t() {}
x64_x64_t::client_t::~client_t() {}
bool x64_x64_t::client_t::valid_impl() { return true; }

x64_x64_t::x64_x64_t(size_t N) : state(nullptr), N(N)
{
  if (N <= 128)
    {
      x64_x64_pair<128> *s = new x64_x64_pair<128>;
      state = static_cast<void *>(s);
    }
}

using ty = x64_x64_pair<128>;

x64_x64_t::~x64_x64_t()
{
  ty *s = static_cast<ty *>(state);
  if (s)
    {
      delete s;
    }
}
x64_x64_t::client_t x64_x64_t::client()
{
  ty *s = static_cast<ty *>(state);
  assert(s);
  client_t res;

  static_assert(sizeof(client_t::state) ==
                    decltype(x64_x64_pair<128>::client)::serialize_size() *
                        sizeof(uint64_t),
                "");
  s->client.serialize(&res.state[0]);
  return res;
}

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

namespace hostrpc
{
thread_local unsigned my_id = 0;
slot_owner tracker;
}  // namespace hostrpc

void hazard(void)
{
  using namespace hostrpc;
  constexpr size_t N = 128;

  x64_x64_pair<N> p;
  (void)p;
#if 0
  _Atomic bool server_live(true);

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
        bool did_work = p.server.rpc_handle(nullptr, &server_location);
        if (did_work)
          {
            count++;
          }
        else
          {
            since_work++;
            platform::sleep_briefly();
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
        if (p.client.rpc_invoke<true>(&scratch))
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
#endif
}
