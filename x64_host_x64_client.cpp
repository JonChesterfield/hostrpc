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
template <typename SZ>
struct x64_x64_pair
{
  x64_x64_client<SZ> client;
  x64_x64_server<SZ> server;
  SZ sz;

  x64_x64_pair(SZ sz) : sz(sz)
  {
    size_t N = sz.N();
    using namespace hostrpc;
    size_t buffer_size = sizeof(page_t) * N;

    // placement new[] requires additional space to wire up delete[]
    hostrpc::page_t *client_buffer = static_cast<page_t *>(
        x64_native::allocate(alignof(page_t), buffer_size));
    hostrpc::page_t *server_buffer = static_cast<page_t *>(
        x64_native::allocate(alignof(page_t), buffer_size));

    for (size_t i = 0; i < N; i++)
      {
        new (client_buffer + i) page_t;
        new (server_buffer + i) page_t;
      }

    assert(client_buffer != server_buffer);

    auto *send_data = x64_allocate_slot_bitmap_data(N);
    auto *recv_data = x64_allocate_slot_bitmap_data(N);
    auto *client_locks_data = x64_allocate_slot_bitmap_data(N);
    auto *server_locks_data = x64_allocate_slot_bitmap_data(N);

    slot_bitmap_all_svm send(N, send_data);
    slot_bitmap_all_svm recv(N, recv_data);
    slot_bitmap_device client_locks(N, client_locks_data);
    slot_bitmap_device server_locks(N, server_locks_data);

    client = {sz, recv, send, client_locks, server_buffer, client_buffer};
    server = {sz, send, recv, server_locks, client_buffer, server_buffer};
  }
  ~x64_x64_pair()
  {
    size_t N = sz.N();

    assert(client.inbox.data() == server.outbox.data());
    assert(client.outbox.data() == server.inbox.data());

    hostrpc::x64_allocate_slot_bitmap_data_deleter del;
    del(client.inbox.data());
    del(server.inbox.data());
    del(client.active.data());
    del(server.active.data());

    assert(client.local_buffer != server.local_buffer);

    for (size_t i = 0; i < N; i++)
      {
        client.local_buffer[i].~page_t();
        server.local_buffer[i].~page_t();
      }

    x64_native::deallocate(client.local_buffer);
    x64_native::deallocate(server.local_buffer);
  }
};

// TODO: Handle N variable w/out loss efficiency
using ty = x64_x64_pair<hostrpc::size_runtime>;

constexpr size_t round(size_t x) { return 64u * ((x + 63u) / 64u); }

static_assert(0 == round(0), "");
static_assert(64 == round(1), "");
static_assert(64 == round(2), "");
static_assert(64 == round(63), "");
static_assert(64 == round(64), "");
static_assert(128 == round(65), "");
static_assert(128 == round(127), "");
static_assert(128 == round(128), "");
static_assert(192 == round(129), "");

x64_x64_t::x64_x64_t(size_t N) : state(nullptr)
{
  N = round(N);
  hostrpc::size_runtime sz(N);
  ty *s = new (std::nothrow) ty(sz);
  state = static_cast<void *>(s);
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
