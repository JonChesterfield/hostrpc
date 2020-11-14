#ifndef HOSTRPC_X64_TARGET_TYPE_HPP_INCLUDED
#define HOSTRPC_X64_TARGET_TYPE_HPP_INCLUDED

#include "allocator.hpp"
#include "base_types.hpp"
#include "detail/client_impl.hpp"
#include "detail/platform_detect.h"
#include "detail/server_impl.hpp"
#include "host_client.hpp"

namespace hostrpc
{
template <int device_num>
struct x64_target_type
{
  using SZ = hostrpc::size_runtime;
  using Copy = copy_functor_given_alias;
  using Word = uint64_t;
  using client_type = client_impl<Word, SZ, Copy, counters::client_nop>;
  using server_type = server_impl<Word, SZ, Copy, counters::server_nop>;

  client_type client;
  server_type server;

  using AllocBuffer =
      hostrpc::allocator::openmp_shared<alignof(page_t), device_num>;
  using AllocInboxOutbox = hostrpc::allocator::openmp_shared<64, device_num>;

  using AllocLocal = hostrpc::allocator::host_libc<64>;
  using AllocRemote = hostrpc::allocator::openmp_target<64, device_num>;

  using storage_type = allocator::store_impl<AllocBuffer, AllocInboxOutbox,
                                             AllocLocal, AllocRemote>;

  storage_type storage;
  HOSTRPC_ANNOTATE x64_target_type(size_t N)
  {
    AllocBuffer alloc_buffer;
    AllocInboxOutbox alloc_inbox_outbox;

    AllocLocal alloc_local;
    AllocRemote alloc_remote;

    SZ sz(N);
    storage = host_client(alloc_buffer, alloc_inbox_outbox, alloc_local,
                          alloc_remote, sz, &server, &client);
  }

  HOSTRPC_ANNOTATE ~x64_target_type() { storage.destroy(); }
  HOSTRPC_ANNOTATE x64_target_type(const x64_target_type &) = delete;

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
