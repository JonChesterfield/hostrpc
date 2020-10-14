#ifndef X64_HOST_PTX_CLIENT_HPP_INCLUDED
#define X64_HOST_PTX_CLIENT_HPP_INCLUDED

#include "detail/client_impl.hpp"
#include "detail/common.hpp"
#include "detail/server_impl.hpp"

#include "memory.hpp"
#include "test_common.hpp"

#if defined(__x86_64__)
#include "memory_cuda.hpp"
#include <cstdio>
#include <cstdlib>
#include <vector>

#endif

namespace hostrpc
{
#if defined(__x86_64__)
template <typename T>
struct x64_ptx_pair
{
  x64_ptx_pair(size_t align, size_t element_count, std::vector<void *> &tofree)
      : x64(nullptr), ptx(nullptr)
  {
    size_t element_size = sizeof(T);
    size_t bytes = element_size * element_count;
    void *p;
    void *void_x64_buffer = cuda::allocate_shared(align, bytes, &p);
    if (void_x64_buffer)
      {
        tofree.push_back(p);
        void *void_ptx_buffer = cuda::device_ptr_from_host_ptr(p);
        if (void_ptx_buffer)
          {
            void_ptx_buffer = cuda::align_pointer_up(void_ptx_buffer, align);
            x64 =
                hostrpc::careful_array_cast<T>(void_x64_buffer, element_count);
            assert(x64);
            ptx =
                hostrpc::careful_array_cast<T>(void_ptx_buffer, element_count);
            assert(ptx);
            return;
          }
      }

    fprintf(stderr, "Failed to construct x64_ptx_pair_T, exit\n");
    exit(1);
  }
  T *x64;
  T *ptx;
};

template <typename T>
T ptx_allocate_slot_bitmap_data_alloc(size_t size, std::vector<void *> &to_free)
{
  constexpr size_t bps = T::bits_per_slot();
  static_assert(bps == 1 || bps == 8, "");
  const size_t align = 64;
  void *p;
  void *memory = cuda::allocate_gpu(align, size * bps, &p);
  if (memory)
    {
      to_free.push_back(p);
    }
  return careful_cast_to_bitmap<T>(memory, size);
}

#endif

template <typename SZ, typename Fill, typename Use, typename Operate,
          typename Clear, typename ClientCounter = counters::client,
          typename ServerCounter = counters::server>
struct x64_ptx_pair_T
{
  using Copy = copy_functor_given_alias;
  using Step = nop_stepper;

  using Word = uint32_t;
  using client_type =
      client_impl<Word, SZ, Copy, Fill, Use, Step, ClientCounter>;
  using server_type =
      server_impl<Word, SZ, Copy, Operate, Clear, Step, ServerCounter>;

  client_type client;
  server_type server;

#if defined(__x86_64__)
  // todo: verify that host/gpu size mismatch is OK here
  std::vector<void *> tofree_gpu;     // todo: array
  std::vector<void *> tofree_shared;  // todo: array
#endif

  x64_ptx_pair_T(SZ sz)
  {
#if defined(__x86_64__)
    size_t N = sz.N();

    x64_ptx_pair<page_t> buffer(alignof(page_t), N, tofree_shared);

    // area read/write by either
    x64_ptx_pair<typename server_type::outbox_t::Ty> send(64, N, tofree_shared);
    x64_ptx_pair<typename server_type::inbox_t::Ty> recv(64, N, tofree_shared);

    // only accessed by client
    auto client_active =
        ptx_allocate_slot_bitmap_data_alloc<typename client_type::lock_t>(
            N, tofree_gpu);
    auto client_staging =
        ptx_allocate_slot_bitmap_data_alloc<typename client_type::staging_t>(
            N, tofree_gpu);

    // only accessed by server
    auto server_active =
        x64_allocate_slot_bitmap_data_alloc<typename server_type::lock_t>(N);

    auto server_staging =
        x64_allocate_slot_bitmap_data_alloc<typename server_type::staging_t>(N);

    // Should check for nullptr here

    server = {sz,         server_active, recv.x64, send.x64, server_staging,
              buffer.x64, buffer.x64};

    client = {sz,         client_active, send.ptx, recv.ptx, client_staging,
              buffer.ptx, buffer.ptx};

    assert(client.size() == N);
    assert(server.size() == N);

#else
    (void)sz;
#endif
  }

  ~x64_ptx_pair_T()
  {
#if defined(__x86_64__)
    size_t N = client.size();
    assert(server.size() == N);

    // can't easily compare host/gpu pointers for aliasing in asserts here

    cuda::deallocate_shared(server.inbox.data());
    cuda::deallocate_shared(server.outbox.data());

    cuda::deallocate_gpu(client.active.data());
    cuda::deallocate_gpu(client.staging.data());

    free(server.active.data());
    free(server.staging.data());

    // buffers alias
    assert(server.local_buffer == server.remote_buffer);
    for (size_t i = 0; i < N; i++)
      {
        server.local_buffer[i].~page_t();
      }
    cuda::deallocate_shared(server.local_buffer);
#endif
  }
};
}  // namespace hostrpc

#endif
