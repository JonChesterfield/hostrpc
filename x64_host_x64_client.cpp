#include "detail/client_impl.hpp"
#include "detail/server_impl.hpp"
#include "interface.hpp"
#include "memory.hpp"
#include "test_common.hpp"
#include <string.h>

#include <new>

namespace hostrpc
{
template <typename SZ>
using x64_x64_client =
    hostrpc::client_indirect_impl<SZ, hostrpc::copy_functor_memcpy_pull,
                                  hostrpc::nop_stepper>;

template <typename SZ>
using x64_x64_server =
    hostrpc::server_indirect_impl<SZ, hostrpc::copy_functor_memcpy_pull,
                                  hostrpc::nop_stepper>;

static _Atomic uint64_t *x64_allocate_atomic_uint64_array(size_t size)
{
  assert(size % 64 == 0 && "Size must be a multiple of 64");
  constexpr const static size_t align = 64;
  void *memory = hostrpc::x64_native::allocate(align, size);
  return hostrpc::careful_array_cast<_Atomic uint64_t>(memory, size);
}

// This doesn't especially care about fill/use/operate/step
// It needs new, probably shouldn't try to compile it on non-x64
template <typename SZ>
struct x64_x64_pair
{
  using client_type = x64_x64_client<SZ>;
  using server_type = x64_x64_server<SZ>;
  client_type client;
  server_type server;
  SZ sz;

  x64_x64_pair(SZ sz) : sz(sz)
  {
    size_t N = sz.N();
    using namespace hostrpc;
    size_t buffer_size = sizeof(page_t) * N;

    hostrpc::page_t *client_buffer = hostrpc::careful_array_cast<page_t>(
        x64_native::allocate(alignof(page_t), buffer_size), N);
    hostrpc::page_t *server_buffer = hostrpc::careful_array_cast<page_t>(
        x64_native::allocate(alignof(page_t), buffer_size), N);

    assert(client_buffer != server_buffer);

    auto *send_data = x64_allocate_atomic_uint64_array(N);
    auto *recv_data = x64_allocate_atomic_uint64_array(N);
    auto *client_locks_data = x64_allocate_atomic_uint64_array(N);
    auto *client_outbox_staging_data = x64_allocate_atomic_uint64_array(N);
    auto *server_locks_data = x64_allocate_atomic_uint64_array(N);
    auto *server_outbox_staging_data = x64_allocate_atomic_uint64_array(N);

    slot_bitmap_all_svm send(N, send_data);
    slot_bitmap_all_svm recv(N, recv_data);
    slot_bitmap_device client_locks(N, client_locks_data);
    slot_bitmap_coarse client_outbox_staging(N, client_outbox_staging_data);
    slot_bitmap_device server_locks(N, server_locks_data);
    slot_bitmap_coarse server_outbox_staging(N, server_outbox_staging_data);

    client = {sz,
              recv,
              send,
              client_locks,
              client_outbox_staging,
              server_buffer,
              client_buffer};
    server = {sz,
              send,
              recv,
              server_locks,
              server_outbox_staging,
              client_buffer,
              server_buffer};
  }
  ~x64_x64_pair()
  {
    size_t N = sz.N();

    assert(client.inbox.data() == server.outbox.data());
    assert(client.outbox.data() == server.inbox.data());

    hostrpc::x64_native::deallocate(client.inbox.data());
    hostrpc::x64_native::deallocate(server.inbox.data());
    hostrpc::x64_native::deallocate(client.active.data());
    hostrpc::x64_native::deallocate(server.active.data());

    hostrpc::x64_native::deallocate(client.outbox_staging.data());
    hostrpc::x64_native::deallocate(server.outbox_staging.data());

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

x64_x64_t::x64_x64_t(size_t N) : state(nullptr)
{
  N = hostrpc::round(N);
  hostrpc::size_runtime sz(N);
  assert(sz.N() != 0);
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

x64_x64_t::client_t x64_x64_t::client()
{
  ty *s = static_cast<ty *>(state);
  assert(s);
  ty::client_type &ct = s->client;
  return {ct};
}

x64_x64_t::server_t x64_x64_t::server()
{
  ty *s = static_cast<ty *>(state);
  assert(s);
  ty::server_type &st = s->server;
  return {st};
}

bool x64_x64_t::client_t::invoke(hostrpc::closure_func_t fill, void *fill_state,
                                 hostrpc::closure_func_t use, void *use_state)
{
  return invoke<ty::client_type>(fill, fill_state, use, use_state);
}

bool x64_x64_t::client_t::invoke_async(hostrpc::closure_func_t fill,
                                       void *fill_state, closure_func_t use,
                                       void *use_state)
{
  return invoke_async<ty::client_type>(fill, fill_state, use, use_state);
}

hostrpc::client_counters x64_x64_t::client_t::get_counters()
{
  return state.open<ty::client_type>()->get_counters();
}

bool x64_x64_t::server_t::handle(hostrpc::closure_func_t func,
                                 void *application_state, uint64_t *l)
{
  return handle<ty::server_type>(func, application_state, l);
}

}  // namespace hostrpc
