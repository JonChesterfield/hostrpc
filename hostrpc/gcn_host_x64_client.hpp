#ifndef GCN_HOST_X64_CLIENT_HPP_INCLUDED
#define GCN_HOST_X64_CLIENT_HPP_INCLUDED

#include "detail/client_impl.hpp"
#include "detail/common.hpp"
#include "detail/server_impl.hpp"

// client parameterised on fill, use
// server parameterised on operate, clear
// All void struct::call(hostrpc::page_t*,void*)

#if defined(__x86_64__)
#include "hsa.h"
#endif

namespace hostrpc
{
template <typename SZ, typename Fill, typename Use, typename Operate,
          typename Clear>
struct gcn_x64_pair_T
{
  using Copy = copy_functor_given_alias;
  using Step = nop_stepper;

  using client_type = client_impl<SZ, Copy, Fill, Use, Step>;
  using server_type = server_impl<SZ, Copy, Operate, Clear, Step>;

  client_type client;
  server_type server;

  gcn_x64_pair_T(SZ sz, uint64_t fine_handle, uint64_t coarse_handle)
  {
    // TODO: This is very similar to x64_host_gcn_client
    // Should be able to abstract over the allocation location

#if defined(__x86_64__)
    size_t N = sz.N();
    hsa_region_t fine = {.handle = fine_handle};
    hsa_region_t coarse = {.handle = coarse_handle};

    page_t *client_buffer = careful_array_cast<page_t>(
        hsa_amdgpu::allocate(fine_handle, alignof(page_t), N * sizeof(page_t)),
        N);

    page_t *server_buffer = client_buffer;

    // could be malloc here, gpu can't see the client locks
    auto send = hsa_allocate_slot_bitmap_data_alloc<message_bitmap>(fine, N);
    auto recv = hsa_allocate_slot_bitmap_data_alloc<message_bitmap>(fine, N);
    auto client_active =
        hsa_allocate_slot_bitmap_data_alloc<lock_bitmap>(fine, N);
    auto client_staging =
        hsa_allocate_slot_bitmap_data_alloc<slot_bitmap_coarse>(fine, N);
    auto server_active =
        hsa_allocate_slot_bitmap_data_alloc<lock_bitmap>(coarse, N);
    auto server_staging =
        hsa_allocate_slot_bitmap_data_alloc<slot_bitmap_coarse>(coarse, N);

    client = {
        sz,           recv, send, client_active, client_staging, server_buffer,
        client_buffer};
    server = {
        sz,           send, recv, server_active, server_staging, client_buffer,
        server_buffer};

    assert(client.size() == N);
    assert(server.size() == N);
#else
    (void)sz;
    (void)fine_handle;
    (void)coarse_handle;
#endif
  }

  ~gcn_x64_pair_T()
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
