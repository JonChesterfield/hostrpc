#include "base_types.hpp"
#include "catch.hpp"
#include "test_common.hpp"

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

#if defined(__x86_64__)

#include "allocator_libc.hpp"
#include "host_client.hpp"
#include "detail/client_impl.hpp"
#include "detail/server_impl.hpp"

namespace hostrpc
{
struct x64_x64_t
{
  using SZ = hostrpc::size_runtime;
  using Copy = copy_functor_given_alias;
  using Step = nop_stepper;
  using Word = uint32_t;
  using client_type = client_impl<Word, SZ, Copy, Step>;
  using server_type = server_impl<Word, SZ, Copy, Step>;


  client_type client;
  server_type server;

  using AllocBuffer = hostrpc::allocator::host_libc<alignof(page_t)>;
  using AllocInboxOutbox = hostrpc::allocator::host_libc<64>;

  using AllocLocal = hostrpc::allocator::host_libc<64>;
  using AllocRemote = hostrpc::allocator::host_libc<64>;

  using storage_type = allocator::store_impl<AllocBuffer, AllocInboxOutbox,
                                             AllocLocal, AllocRemote>;

  storage_type storage;
  
  // This probably can't be copied, but could be movable
  x64_x64_t(size_t minimum_number_slots)
  {
    size_t N = hostrpc::round64(minimum_number_slots);
    AllocBuffer alloc_buffer;
    AllocInboxOutbox alloc_inbox_outbox;
    AllocLocal alloc_local;
    AllocRemote alloc_remote;

    SZ sz(N);
    storage = host_client(alloc_buffer, alloc_inbox_outbox, alloc_local,
                          alloc_remote, sz, &server, &client);

    
  }

  ~x64_x64_t() {storage.destroy();}

  x64_x64_t(const x64_x64_t &) = delete;
  bool valid()
  { /*todo*/
    return true;
  }  // true if construction succeeded

  template <bool have_continuation>
  bool rpc_invoke(void *fill, void *use) noexcept
  {
    return client
        .rpc_invoke<indirect::fill, indirect::use, have_continuation>(fill,
                                                                      use);
  }

  bool rpc_handle(void *operate_state, void *clear_state) noexcept
  {
    return server.rpc_handle<indirect::operate, indirect::clear>(
        operate_state, clear_state);
  }

  bool rpc_handle(void *operate_state, void *clear_state,
                  uint32_t *location_arg) noexcept
  {
    return server.rpc_handle<indirect::operate, indirect::clear>(
        operate_state, clear_state, location_arg);
  }

  client_counters client_counters() { return client.get_counters(); }


};
}  // namespace hostrpc
#endif

namespace hostrpc
{
thread_local unsigned my_id = 0;
}  // namespace hostrpc

TEST_CASE("x64_x64_stress")
{
  using namespace hostrpc;
  hostrpc::x64_x64_t p(100);

  auto op_func = [](hostrpc::page_t *page) {
    for (unsigned c = 0; c < 64; c++)
      {
        hostrpc::cacheline_t &line = page->cacheline[c];
#if 0
        std::swap(line.element[0], line.element[7]);
        std::swap(line.element[1], line.element[6]);
        std::swap(line.element[2], line.element[5]);
        std::swap(line.element[3], line.element[4]);
#endif
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
    my_id = id;
    unsigned count = 0;

    uint32_t server_location = 0;
    hostrpc::closure_pair op_arg = make_closure_pair(&op_func);
    hostrpc::closure_pair cl_arg = make_closure_pair(&cl_func);
    for (;;)
      {
        if (!server_live)
          {
            printf("server %u did %u tasks\n", id, count);
            break;
          }
        bool did_work =
            p.rpc_handle(static_cast<void *>(&op_arg),
                         static_cast<void *>(&cl_arg), &server_location);
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
    my_id = id;
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
    hostrpc::closure_pair fill_cp = make_closure_pair(&fill);
    hostrpc::closure_pair use_cp = make_closure_pair(&use);

    for (unsigned r = 0; r < reps; r++)
      {
        init_page(&scratch, id + r);
        init_page(&expect, id + r + 1);

        if (p.rpc_invoke<true>(&fill_cp, &use_cp))
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
