#ifndef X64_HOST_GCN_CLIENT_HPP_INCLUDED
#define X64_HOST_GCN_CLIENT_HPP_INCLUDED

#include "detail/client_impl.hpp"
#include "detail/common.hpp"
#include "detail/server_impl.hpp"

#include "memory.hpp"
#include "test_common.hpp"

#if defined(__x86_64__)
// May want this header to be free of hsa stuff, following the nvptx layout
#include "hsa.h"
#endif

namespace hostrpc
{
template <typename SZ, typename Fill, typename Use, typename Operate,
          typename Clear>
struct x64_gcn_pair_T
{
  using Copy = copy_functor_given_alias;
  using Step = nop_stepper;

  using Word = uint64_t;
  using client_type =
      client_impl<Word, SZ, Copy, Fill, Use, Step, counters::client>;
  using server_type =
      server_impl<Word, SZ, Copy, Operate, Clear, Step, counters::server>;

  client_type client;
  server_type server;

  x64_gcn_pair_T(SZ sz, uint64_t fine_handle, uint64_t coarse_handle)
  {
#if defined(__x86_64__)
    size_t N = sz.N();
    hsa_region_t fine = {.handle = fine_handle};
    hsa_region_t coarse = {.handle = coarse_handle};

    // Shared buffer. todo: drop the redundant pointer
    hostrpc::page_t *client_buffer = hostrpc::careful_array_cast<page_t>(
        hostrpc::hsa_amdgpu::allocate(fine_handle, alignof(page_t),
                                      N * sizeof(page_t)),
        N);

    hostrpc::page_t *server_buffer =
        client_buffer;

    // allocating in coarse is probably not sufficient, likely to need to mark
    // the pointer with an address space
    // server_active could be 'malloc', gcn can't access it

    // fine grained area, can read/write from either client or server
    // todo: send/recv in terms of server type instead of client?
    auto send =
        hsa_allocate_slot_bitmap_data_alloc<typename client_type::outbox_t>(
            fine, N);
    auto recv =
        hsa_allocate_slot_bitmap_data_alloc<typename client_type::inbox_t>(fine,
                                                                           N);

    // only accessed by client
    auto client_active =
        hsa_allocate_slot_bitmap_data_alloc<typename client_type::lock_t>(
            coarse, N);
    auto client_staging =
        hsa_allocate_slot_bitmap_data_alloc<typename client_type::staging_t>(
            coarse, N);

    // only accessed by server
    auto server_active =
        x64_allocate_slot_bitmap_data_alloc<typename server_type::lock_t>(N);

    auto server_staging =
        x64_allocate_slot_bitmap_data_alloc<typename server_type::staging_t>(N);

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
    free(server.active.data());
    free(server.staging.data());

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
