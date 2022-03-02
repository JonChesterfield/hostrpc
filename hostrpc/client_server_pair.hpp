#ifndef HOSTRPC_CLIENT_SERVER_PAIR_HPP_INCLUDED
#define HOSTRPC_CLIENT_SERVER_PAIR_HPP_INCLUDED

#include "base_types.hpp"

#include "allocator.hpp"
#include "detail/client_impl.hpp"
#include "detail/server_impl.hpp"
#include "host_client.hpp"
#include "platform/detect.hpp"

// local / remote distinction may not be useful here
// allocator is defined in terms of shared memory, buffers on the device doing
// the allocation and buffers on the other device. That is, it doesn't have a
// client/server notion, just 'here' and 'there' This probably needs to take
// AllocClient and AllocRemote parameters.

namespace hostrpc
{
template <
    typename SZ_,    // size_compiletime or size_runtime, number of slots
    typename Word_,  // width of atomic operations (strictly could be different
                     // on each side)
    typename AllocBuffer_,  // allocate shared memory used for argument passing
    typename AllocInboxOutbox_,  // shared memory use for state transitions
    typename AllocLocal_, typename AllocRemote_,
    typename client_counter = counters::client_nop,
    typename server_counter = counters::server_nop>
struct client_server_pair_t
{
  using SZ = SZ_;
  using Word = Word_;
  using AllocBuffer = AllocBuffer_;
  using AllocInboxOutbox = AllocInboxOutbox_;
  using AllocLocal = AllocLocal_;
  using AllocRemote = AllocRemote_;

  using client_type = client<Word, SZ, client_counter>;
  using server_type = server<Word, SZ, server_counter>;

  using storage_type = allocator::store_impl<AllocBuffer, AllocInboxOutbox,
                                             AllocLocal, AllocRemote>;

  client_type client;
  server_type server;
  storage_type storage;

  HOSTRPC_ANNOTATE client_server_pair_t(const client_server_pair_t &) = delete;
  HOSTRPC_ANNOTATE client_server_pair_t &operator=(
      const client_server_pair_t &) = delete;

  HOSTRPC_ANNOTATE client_server_pair_t(client_server_pair_t &&) = default;
  HOSTRPC_ANNOTATE client_server_pair_t &operator=(client_server_pair_t &&) =
      default;

  HOSTRPC_ANNOTATE client_server_pair_t() {}

  HOSTRPC_ANNOTATE client_server_pair_t(SZ sz, AllocBuffer alloc_buffer,
                                        AllocInboxOutbox alloc_inbox_outbox,
                                        AllocLocal alloc_local,
                                        AllocRemote alloc_remote)
      // host_client is mapping &server to local and &client to remote
      : storage(host_client(alloc_buffer, alloc_inbox_outbox, alloc_local,
                            alloc_remote, sz, &server, &client))
  {
  }

  HOSTRPC_ANNOTATE ~client_server_pair_t() { storage.destroy(); }

  HOSTRPC_ANNOTATE bool valid() { return storage.valid(); }
  HOSTRPC_ANNOTATE client_counters client_counters()
  {
    return client.get_counters();
  }
  HOSTRPC_ANNOTATE server_counters server_counters()
  {
    return server.get_counters();
  }

  HOSTRPC_ANNOTATE void dump()
  {
#if HOSTRPC_HAVE_STDIO
    fprintf(stderr, "storage:\n");
    storage.dump();
    fprintf(stderr, "server:\n");
    server.dump();
    fprintf(stderr, "client:\n");
    client.dump();
#endif
  }
};
}  // namespace hostrpc
#endif
