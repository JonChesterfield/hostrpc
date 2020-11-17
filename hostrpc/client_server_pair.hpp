#ifndef HOSTRPC_CLIENT_SERVER_PAIR_HPP_INCLUDED
#define HOSTRPC_CLIENT_SERVER_PAIR_HPP_INCLUDED

#include "base_types.hpp"

#include "allocator.hpp"
#include "detail/client_impl.hpp"
#include "detail/platform_detect.h"
#include "detail/server_impl.hpp"
#include "host_client.hpp"

namespace hostrpc
{
template <typename SZ_, typename Copy_, typename Word_, typename AllocBuffer_,
          typename AllocInboxOutbox_, typename AllocLocal_,
          typename AllocRemote_>
struct client_server_pair_t
{
  using SZ = SZ_;
  using Copy = Copy_;
  using Word = Word_;
  using AllocBuffer = AllocBuffer_;
  using AllocInboxOutbox = AllocInboxOutbox_;
  using AllocLocal = AllocLocal_;
  using AllocRemote = AllocRemote_;

  using client_type = client_impl<Word, SZ, Copy, counters::client_nop>;
  using server_type = server_impl<Word, SZ, Copy, counters::server_nop>;

  using storage_type = allocator::store_impl<AllocBuffer, AllocInboxOutbox,
                                             AllocLocal, AllocRemote>;

  client_type client;
  server_type server;
  storage_type storage;

  HOSTRPC_ANNOTATE client_server_pair_t(SZ sz, AllocBuffer alloc_buffer,
                                        AllocInboxOutbox alloc_inbox_outbox,
                                        AllocLocal alloc_local,
                                        AllocRemote alloc_remote)
  {
    storage = host_client(alloc_buffer, alloc_inbox_outbox, alloc_local,
                          alloc_remote, sz, &server, &client);
  }

  HOSTRPC_ANNOTATE ~client_server_pair_t() { storage.destroy(); }
  HOSTRPC_ANNOTATE client_server_pair_t(const client_server_pair_t &) = delete;

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
