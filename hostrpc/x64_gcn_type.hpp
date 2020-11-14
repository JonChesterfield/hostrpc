#ifndef HOSTRPC_X64_GCN_TYPE_HPP_INCLUDED
#define HOSTRPC_X64_GCN_TYPE_HPP_INCLUDED

#include "base_types.hpp"

#include "allocator.hpp"
#include "detail/client_impl.hpp"
#include "detail/platform_detect.h"
#include "detail/server_impl.hpp"
#include "host_client.hpp"

namespace hostrpc
{
struct x64_gcn_type
{
  using SZ = hostrpc::size_runtime;
  using Copy = copy_functor_given_alias;
  using Word = uint64_t;
  using client_type = client_impl<Word, SZ, Copy, counters::client_nop>;
  using server_type = server_impl<Word, SZ, Copy, counters::server_nop>;

  client_type client;
  server_type server;

  using AllocBuffer = hostrpc::allocator::hsa<alignof(page_t)>;
  using AllocInboxOutbox = hostrpc::allocator::hsa<64>;

  using AllocLocal = hostrpc::allocator::host_libc<64>;
  using AllocRemote = hostrpc::allocator::hsa<64>;

  using storage_type = allocator::store_impl<AllocBuffer, AllocInboxOutbox,
                                             AllocLocal, AllocRemote>;

  storage_type storage;
  HOSTRPC_ANNOTATE x64_gcn_type(size_t N, uint64_t hsa_region_t_fine_handle,
                                uint64_t hsa_region_t_coarse_handle)
  {
    uint64_t fine_handle = hsa_region_t_fine_handle;
    uint64_t coarse_handle = hsa_region_t_coarse_handle;

    AllocBuffer alloc_buffer(fine_handle);
    AllocInboxOutbox alloc_inbox_outbox(fine_handle);

    AllocLocal alloc_local;
    AllocRemote alloc_remote(coarse_handle);

    SZ sz(N);
    storage = host_client(alloc_buffer, alloc_inbox_outbox, alloc_local,
                          alloc_remote, sz, &server, &client);
  }

  HOSTRPC_ANNOTATE ~x64_gcn_type() { storage.destroy(); }
  HOSTRPC_ANNOTATE x64_gcn_type(const x64_gcn_type &) = delete;

  HOSTRPC_ANNOTATE client_counters client_counters()
  {
    return client.get_counters();
  }
  HOSTRPC_ANNOTATE server_counters server_counters()
  {
    return server.get_counters();
  }
};
}  // namespace hostrpc

#endif
