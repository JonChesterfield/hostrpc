#ifndef X64_HOST_GCN_CLIENT_HPP_INCLUDED
#define X64_HOST_GCN_CLIENT_HPP_INCLUDED

#include "detail/client_impl.hpp"
//#include "detail/common.hpp"
#include "detail/server_impl.hpp"

//#include "memory.hpp"
//#include "test_common.hpp"

#include "allocator_hsa.hpp"
#include "allocator_libc.hpp"
#include "host_client.hpp"

namespace hostrpc
{
template <typename SZ, typename ClientCounter = counters::client,
          typename ServerCounter = counters::server>
struct x64_gcn_pair_T
{
  using Copy = copy_functor_given_alias;
  using Word = uint64_t;

  using client_type = client_impl<Word, SZ, Copy, ClientCounter>;
  using server_type = server_impl<Word, SZ, Copy, ServerCounter>;

  client_type client;
  server_type server;

  using AllocBuffer = hostrpc::allocator::hsa<alignof(page_t)>;
  using AllocInboxOutbox = hostrpc::allocator::hsa<64>;

  using AllocLocal = hostrpc::allocator::host_libc<64>;
  using AllocRemote = hostrpc::allocator::hsa<64>;

  using storage_type = allocator::store_impl<AllocBuffer, AllocInboxOutbox,
                                             AllocLocal, AllocRemote>;

  storage_type storage;
  x64_gcn_pair_T(SZ sz, uint64_t fine_handle, uint64_t coarse_handle)
  {
#if defined(__x86_64__)

    auto alloc_buffer = AllocBuffer(fine_handle);
    auto alloc_inbox_outbox = AllocInboxOutbox(fine_handle);

    auto alloc_local = AllocLocal();
    auto alloc_remote = AllocRemote(coarse_handle);

    storage = host_client(alloc_buffer, alloc_inbox_outbox, alloc_local,
                               alloc_remote, sz, &server, &client);
#else
    (void)sz;
    (void)fine_handle;
    (void)coarse_handle;
#endif
  }

  ~x64_gcn_pair_T()
  {
#if defined(__x86_64__)
    storage.destroy();
#endif
  }
};

}  // namespace hostrpc

#endif
