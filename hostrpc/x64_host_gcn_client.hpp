#ifndef X64_HOST_GCN_CLIENT_HPP_INCLUDED
#define X64_HOST_GCN_CLIENT_HPP_INCLUDED

#include "detail/client_impl.hpp"
#include "detail/common.hpp"
#include "detail/server_impl.hpp"

#include "memory.hpp"
#include "test_common.hpp"

#if defined(__x86_64__)
#include "hsa.h"
#endif

namespace hostrpc
{
struct copy_functor_x64_gcn
    : public hostrpc::copy_functor_interface<copy_functor_x64_gcn>
{
  friend struct hostrpc::copy_functor_interface<copy_functor_x64_gcn>;

  // attempting to move incrementally to a gpu-local buffer to avoid
  // compiler generated accesses to flat memory
  static void push_from_client_to_server_impl(hostrpc::page_t *dst,
                                              const hostrpc::page_t *src)
  {
    // src is coarse memory, dst is fine
    assert(src == dst);
    (void)src;
    (void)dst;
  }

  static void pull_to_client_from_server_impl(hostrpc::page_t *dst,
                                              const hostrpc::page_t *src)
  {
    // dst is coarse memory, src is fine
    assert(src == dst);
    (void)src;
    (void)dst;
  }

  // No copies done by the x64 server as it can't see the gcn local buffer
  static void push_from_server_to_client_impl(hostrpc::page_t *dst,
                                              const hostrpc::page_t *src)
  {
    assert(src == dst);
    (void)src;
    (void)dst;
  }
  static void pull_to_server_from_client_impl(hostrpc::page_t *dst,
                                              const hostrpc::page_t *src)
  {
    assert(src == dst);
    (void)src;
    (void)dst;
  }
};

template <typename SZ, typename Fill, typename Use, typename Operate,
          typename Clear>
struct x64_gcn_pair_T
{
  using Copy = copy_functor_x64_gcn;
  using Step = nop_stepper;

  using client_type = client_impl<SZ, Copy, Fill, Use, Step, counters::client>;
  using server_type = server_impl<SZ, Copy, Operate, Clear, Step>;

  client_type client;
  server_type server;

  x64_gcn_pair_T(SZ sz, uint64_t fine_handle, uint64_t coarse_handle)
  {
#if defined(__x86_64__)
    size_t N = sz.N();
    hsa_region_t fine = {.handle = fine_handle};
    hsa_region_t coarse = {.handle = coarse_handle};

    hostrpc::page_t *client_buffer = hostrpc::careful_array_cast<page_t>(
        hostrpc::hsa_amdgpu::allocate(fine_handle, alignof(page_t),
                                      N * sizeof(page_t)),
        N);

    hostrpc::page_t *server_buffer = client_buffer;

    // allocating in coarse is probably not sufficient, likely to need to mark
    // the pointer with an address space
    // server_active could be 'malloc', gcn can't access it
    auto send =
        hsa_allocate_slot_bitmap_data_alloc<typename client_type::outbox_t>(
            fine, N);
    auto recv =
        hsa_allocate_slot_bitmap_data_alloc<typename client_type::inbox_t>(fine,
                                                                           N);
    auto client_active =
        hsa_allocate_slot_bitmap_data_alloc<lock_bitmap>(coarse, N);
    auto client_staging =
        hsa_allocate_slot_bitmap_data_alloc<typename client_type::staging_t>(
            coarse, N);
    auto server_active =
        hsa_allocate_slot_bitmap_data_alloc<lock_bitmap>(fine, N);
    auto server_staging =
        hsa_allocate_slot_bitmap_data_alloc<typename server_type::staging_t>(
            fine, N);

    client = {sz,           client_active,  recv,
              send,         client_staging, server_buffer,
              client_buffer};

    server = {sz,           server_active,  send,
              recv,         server_staging, client_buffer,
              server_buffer};

    assert(client.size() == N);
    assert(server.size() == N);
#else
    (void)sz;
    (void)fine_handle;
    (void)coarse_handle;
#endif
  }

  ~x64_gcn_pair_T()
  {
#if defined(__x86_64__)
    size_t N = client.size();
    assert(server.size() == N);

    assert(client.inbox.data() == server.outbox.data());
    assert(client.outbox.data() == server.inbox.data());

    hsa_allocate_slot_bitmap_data_free(client.inbox.data());
    hsa_allocate_slot_bitmap_data_free(client.outbox.data());
    hsa_allocate_slot_bitmap_data_free(client.active.data());
    hsa_allocate_slot_bitmap_data_free(client.staging.data());
    hsa_allocate_slot_bitmap_data_free(server.active.data());
    hsa_allocate_slot_bitmap_data_free(server.staging.data());

    // precondition of structure
    assert(client.local_buffer == server.remote_buffer);
    assert(client.remote_buffer == server.local_buffer);

    // postcondition of this instance
    assert(client.local_buffer == client.remote_buffer);
    for (size_t i = 0; i < N; i++)
      {
        client.local_buffer[i].~page_t();
      }
    hsa_memory_free(client.local_buffer);
#endif
  }
};

}  // namespace hostrpc

#endif
