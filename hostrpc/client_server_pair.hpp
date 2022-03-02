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

namespace detail
{
template <typename X, typename Y, bool cmp>
struct select_type;

template <typename X, typename Y>
struct select_type<X, Y, true>
{
  using type = X;
};

template <typename X, typename Y>
struct select_type<X, Y, false>
{
  using type = Y;
};
}  // namespace detail

namespace arch
{

// Four types of buffer involved.
// (page_t aligned) large buffer shared for argument passing
// shared memory used for state transitions
// client local memory that server can't access
// server local memory that client can't access

// Word size and local allocator type for each arch
struct x64
{
  using word_type = uint64_t;
  template <size_t align>
  using allocator = allocator::host_libc<align>;
};

struct gcn
{
  using word_type = uint64_t;
  template <size_t align>
  using allocator = allocator::hsa<align>;
};

struct ptx
{
  using word_type = uint32_t;
  template <size_t align>
  using allocator = allocator::cuda_gpu<align>;
};

template <int device_num>
struct openmp_target
{
  using word_type = uint32_t;  // TODO: Don't really know, 32 is conservative
  template <size_t align>
  using allocator = allocator::openmp_device<align, device_num>;
};

// shared memory allocator type for pairs of architectures

template <typename X, typename Y>
struct pair;

// commutative
template <typename X, typename Y>
struct pair : public pair<Y, X>
{
};

template <>
struct pair<x64, x64>
{
  template <size_t align>
  using allocator = x64::allocator<align>;
};

template <>
struct pair<x64, ptx>
{
  template <size_t align>
  using allocator = allocator::cuda_shared<align>;
};

template <>
struct pair<x64, gcn>
{
  template <size_t align>
  using allocator = allocator::hsa<align>;
};

template <int device_num>
struct pair<x64, openmp_target<device_num>>
{
  template <size_t align>
  using allocator = allocator::openmp_shared<align>;
};

template <typename Local, typename Remote>
struct allocators
{
  using word_type = typename detail::select_type<
      Local, Remote,
      sizeof(typename Local::word_type) <
          sizeof(typename Remote::word_type)>::type::word_type;

  template <size_t align>
  using local_allocator = typename Local::template allocator<align>;

  template <size_t align>
  using remote_allocator = typename Remote::template allocator<align>;

  template <size_t align>
  using shared_allocator =
      typename pair<Local, Remote>::template allocator<align>;
};

}  // namespace arch

template <typename SZ_, typename ClientArch_, typename ServerArch_,
          typename client_counter = counters::client_nop,
          typename server_counter = counters::server_nop>
struct client_server_pair_t
{
  using SZ = SZ_;

  using Allocators = arch::allocators<ClientArch_, ServerArch_>;
  using Word = typename Allocators::word_type;
  using AllocBuffer =
      typename Allocators::template shared_allocator<alignof(page_t)>;
  using AllocInboxOutbox = typename Allocators::template shared_allocator<64>;
  using AllocLocal = typename Allocators::template local_allocator<64>;
  using AllocRemote = typename Allocators::template remote_allocator<64>;

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
