#include "client_impl.hpp"
#include "interface.hpp"
#include "memory.hpp"
#include "server_impl.hpp"

#include <string.h>

#include <new>

namespace hostrpc
{
namespace x64_host_x64_client
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
  static void call(hostrpc::page_t *page, void *dv)
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

}  // namespace x64_host_x64_client

template <typename SZ>
using x64_x64_client = hostrpc::client_impl<
    SZ, hostrpc::copy_functor_memcpy_pull, hostrpc::x64_host_x64_client::fill,
    hostrpc::x64_host_x64_client::use, hostrpc::nop_stepper>;

template <typename SZ>
using x64_x64_server =
    hostrpc::server_impl<SZ, hostrpc::copy_functor_memcpy_pull,
                         hostrpc::x64_host_x64_client::operate,
                         hostrpc::nop_stepper>;

// This doesn't especially care about fill/use/operate/step
// It needs new, probably shouldn't try to compile it on non-x64
template <size_t N>
struct x64_x64_pair
{
  using SZ = size_compiletime<N>;
  x64_x64_client<SZ> client;
  x64_x64_server<SZ> server;

  hostrpc::page_t *client_buffer;
  hostrpc::page_t *server_buffer;
  x64_x64_pair()
  {
    using namespace hostrpc;
    size_t buffer_size = sizeof(page_t) * N;

    // TODO: strictly should placement new here
    client_buffer = reinterpret_cast<page_t *>(
        x64_native::allocate(alignof(page_t), buffer_size));
    server_buffer = reinterpret_cast<page_t *>(
        x64_native::allocate(alignof(page_t), buffer_size));
    assert(client_buffer != server_buffer);

    auto *send_data = x64_allocate_slot_bitmap_data(N);
    auto *recv_data = x64_allocate_slot_bitmap_data(N);
    auto *client_locks_data = x64_allocate_slot_bitmap_data(N);
    auto *server_locks_data = x64_allocate_slot_bitmap_data(N);

    slot_bitmap_all_svm<SZ> send(send_data, SZ{N});
    slot_bitmap_all_svm<SZ> recv(recv_data, SZ{N});
    slot_bitmap_device<SZ> client_locks(client_locks_data, SZ{N});
    slot_bitmap_device<SZ> server_locks(server_locks_data, SZ{N});

    client = {recv, send, client_locks, server_buffer, client_buffer};
    server = {send, recv, server_locks, client_buffer, server_buffer};
  }
  ~x64_x64_pair()
  {
    assert(client.inbox.data() == server.outbox.data());
    assert(client.outbox.data() == server.inbox.data());

    hostrpc::x64_allocate_slot_bitmap_data_deleter del;
    del(client.inbox.data());
    del(server.inbox.data());
    del(client.active.data());
    del(server.active.data());

    assert(client.local_buffer != server.local_buffer);
    x64_native::deallocate(client.local_buffer);
    x64_native::deallocate(server.local_buffer);
  }
};

// TODO: Handle N variable w/out loss efficiency
using ty = x64_x64_pair<128>;

x64_x64_t::x64_x64_t(size_t N) : state(nullptr)
{
  if (N <= 128)
    {
      ty *s = new (std::nothrow) ty;
      state = static_cast<void *>(s);
    }
}

x64_x64_t::~x64_x64_t()
{
  ty *s = static_cast<ty *>(state);
  if (s)
    {
      delete s;
    }
}

bool x64_x64_t::valid() { return state != nullptr; }

static decltype(ty::client) *open_client(uint64_t *state)
{
  return reinterpret_cast<decltype(ty::client) *>(state);
}
static decltype(ty::server) *open_server(uint64_t *state)
{
  return reinterpret_cast<decltype(ty::server) *>(state);
}

x64_x64_t::client_t x64_x64_t::client()
{
  ty *s = static_cast<ty *>(state);
  assert(s);
  client_t res;
  auto *cl = reinterpret_cast<decltype(ty::client) *>(&res.state[0]);
  *cl = s->client;
  return res;
}

__attribute__((used)) x64_x64_t::server_t x64_x64_t::server()
{
  ty *s = static_cast<ty *>(state);
  assert(s);
  server_t res;
  auto *cl = reinterpret_cast<decltype(ty::server) *>(&res.state[0]);
  *cl = s->server;
  return res;
}

bool x64_x64_t::client_t::invoke_impl(void *application_state)
{
  auto *cl = open_client(&state[0]);
  return cl->rpc_invoke<true>(application_state);
}

bool x64_x64_t::client_t::invoke_async_impl(void *application_state)
{
  auto *cl = open_client(&state[0]);
  return cl->rpc_invoke<false>(application_state);
}

bool x64_x64_t::server_t::handle_impl(void *application_state, uint64_t *l)
{
  auto *se = open_server(&state[0]);
  return se->rpc_handle(application_state, l);
}

}  // namespace hostrpc
