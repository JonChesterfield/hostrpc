#ifndef X64_HOST_X64_CLIENT_HPP_INCLUDED
#define X64_HOST_X64_CLIENT_HPP_INCLUDED

#include "detail/client_impl.hpp"
#include "detail/common.hpp"
#include "detail/server_impl.hpp"
#include "memory.hpp"

namespace hostrpc
{
template <typename T>
static T x64_alloc(size_t size)
{
  constexpr size_t bps = T::bits_per_slot();
  static_assert(bps == 1 || bps == 8, "");
  assert(size % 64 == 0 && "Size must be a multiple of 64");
  constexpr const static size_t align = 64;
  void *memory = hostrpc::x64_native::allocate(align, size * bps);
  _Atomic(uint64_t) *m =
      hostrpc::careful_array_cast<_Atomic(uint64_t)>(memory, size * bps);
  return {m};
}

template <typename SZ, typename Fill, typename Use, typename Operate,
          typename Clear>
struct x64_x64_pair_T
{
  using Copy = copy_functor_memcpy_pull;
  using Step = nop_stepper;

  using client_type = client_impl<SZ, Copy, Fill, Use, Step>;
  using server_type = server_impl<SZ, Copy, Operate, Clear, Step>;

  client_type client;
  server_type server;

  x64_x64_pair_T(SZ sz)
  {
    size_t N = sz.N();
    size_t buffer_size = sizeof(page_t) * N;

    hostrpc::page_t *client_buffer = hostrpc::careful_array_cast<page_t>(
        x64_native::allocate(alignof(page_t), buffer_size), N);
    hostrpc::page_t *server_buffer = hostrpc::careful_array_cast<page_t>(
        x64_native::allocate(alignof(page_t), buffer_size), N);

    assert(client_buffer != server_buffer);

    auto send = x64_alloc<message_bitmap>(N);
    auto recv = x64_alloc<message_bitmap>(N);
    auto client_locks = x64_alloc<lock_bitmap>(N);
    auto client_staging = x64_alloc<slot_bitmap_coarse>(N);
    auto server_locks = x64_alloc<lock_bitmap>(N);
    auto server_staging = x64_alloc<slot_bitmap_coarse>(N);

    client = {sz,           recv,           send,
              client_locks, client_staging, server_buffer,
              client_buffer};
    server = {sz,           send,           recv,
              server_locks, server_staging, client_buffer,
              server_buffer};

    assert(client.size() == N);
    assert(server.size() == N);
  }

  ~x64_x64_pair_T()
  {
    size_t N = client.size();
    assert(server.size() == N);

    assert(client.inbox.data() == server.outbox.data());
    assert(client.outbox.data() == server.inbox.data());

    hostrpc::x64_native::deallocate(client.inbox.data());
    hostrpc::x64_native::deallocate(server.inbox.data());
    hostrpc::x64_native::deallocate(client.active.data());
    hostrpc::x64_native::deallocate(server.active.data());

    hostrpc::x64_native::deallocate(client.staging.data());
    hostrpc::x64_native::deallocate(server.staging.data());

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

}  // namespace hostrpc

#endif
