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
  using local_allocator_t = allocator::host_libc<align>;

  x64() {}

  template <size_t align>
  local_allocator_t<align> local_allocator()
  {
    return {};
  }
};

struct gcn
{
  using word_type = uint64_t;
  template <size_t align>
  using local_allocator_t = allocator::hsa<align>;
  template <size_t align>
  using shared_allocator_t = allocator::hsa<align>;

  gcn(uint64_t fine_handle_, uint64_t coarse_handle_)
      : fine_handle(fine_handle_), coarse_handle(coarse_handle_)
  {
  }

  template <size_t align>
  local_allocator_t<align> local_allocator()
  {
    return {coarse_handle};
  }

  template <size_t align>
  shared_allocator_t<align> shared_allocator()
  {
    return {fine_handle};
  }

 private:
  uint64_t fine_handle;
  uint64_t coarse_handle;
};

struct ptx
{
  using word_type = uint32_t;
  template <size_t align>
  using local_allocator_t = hostrpc::allocator::cuda_gpu<align>;
  template <size_t align>
  using shared_allocator_t = hostrpc::allocator::cuda_shared<align>;

  ptx() {}

  template <size_t align>
  local_allocator_t<align> local_allocator()
  {
    return {};
  }

  template <size_t align>
  shared_allocator_t<align> shared_allocator()
  {
    return {};
  }
};

#ifdef _OPENMP
template <int device_num>
struct openmp_target
{
  using word_type = uint32_t;  // TODO: Don't really know, 32 is conservative
  template <size_t align>
  using local_allocator_t = allocator::openmp_device<align, device_num>;
  template <size_t align>
  using shared_allocator_t = allocator::openmp_shared<align, device_num>;

  openmp_target() {}

  template <size_t align>
  local_allocator_t<align> local_allocator()
  {
    return {};
  }

  template <size_t align>
  shared_allocator_t<align> shared_allocator()
  {
    return {};
  }
};
#endif
  
// shared memory allocator type for pairs of architectures

template <typename X, typename Y>
struct pair;

// commutative
template <typename X, typename Y>
struct pair : public pair<Y, X>
{
  pair(X x, Y y) : pair<Y, X>(y, x) {}
};

// client and server on same arch can use local allocator
template <typename T>
struct pair<T, T>
{
  pair(T s_) : s(s_) {}
  pair(T s_, T) : s(s_) {}

  template <size_t align>
  using shared_allocator_t = typename T::template local_allocator_t<align>;

  template <size_t align>
  shared_allocator_t<align> shared_allocator()
  {
    return s.template local_allocator<align>();
  }

 private:
  T s;
};

template <>
struct pair<x64, ptx>
{
  pair(x64 x_, ptx y_) : x(x_), y(y_) {}

  template <size_t align>
  using shared_allocator_t = allocator::cuda_shared<align>;

  template <size_t align>
  shared_allocator_t<align> shared_allocator()
  {
    return y.template shared_allocator<align>();
  }

 private:
  x64 x;
  ptx y;
};

template <>
struct pair<x64, gcn>
{
  pair(x64 x_, gcn y_) : x(x_), y(y_) {}

  template <size_t align>
  using shared_allocator_t = allocator::hsa<align>;

  template <size_t align>
  shared_allocator_t<align> shared_allocator()
  {
    return y.template shared_allocator<align>();
  }

 private:
  x64 x;
  gcn y;
};

#ifdef _OPENMP
  template <int device_num>
struct pair<x64, openmp_target<device_num>>
{
  pair(x64 x_, openmp_target<device_num> y_) : x(x_), y(y_) {}

  template <size_t align>
  using shared_allocator_t = allocator::openmp_shared<align, device_num>;

  template <size_t align>
  shared_allocator_t<align> shared_allocator()
  {
    return y.template shared_allocator<align>();
  }

 private:
  x64 x;
  openmp_target<device_num> y;
};
#endif

template <typename Local, typename Remote>
struct allocators
{
  using word_type = typename detail::select_type<
      Local, Remote,
      sizeof(typename Local::word_type) <
          sizeof(typename Remote::word_type)>::type::word_type;

  allocators(Local l_, Remote r_) : l(l_), r(r_) {}

  template <size_t align>
  using local_allocator_t = typename Local::template local_allocator_t<align>;

  template <size_t align>
  local_allocator_t<align> local_allocator()
  {
    return l.template local_allocator<align>();
  }

  template <size_t align>
  using remote_allocator_t = typename Remote::template local_allocator_t<align>;

  template <size_t align>
  remote_allocator_t<align> remote_allocator()
  {
    return r.template local_allocator<align>();
  }

  template <size_t align>
  using shared_allocator_t =
      typename pair<Local, Remote>::template shared_allocator_t<align>;

  template <size_t align>
  shared_allocator_t<align> shared_allocator()
  {
    pair<Local, Remote> p(l, r);
    return p.template shared_allocator<align>();
  }

 private:
  Local l;
  Remote r;
};

}  // namespace arch

template <typename SZ_, typename ClientArch_, typename ServerArch_>
struct client_server_pair_t
{
  using SZ = SZ_;
  using ClientArch = ClientArch_;
  using ServerArch = ServerArch_;

  using Allocators = arch::allocators<ClientArch_, ServerArch_>;
  using Word = typename Allocators::word_type;

  using BufferElement = page_t;
  
  using AllocBuffer =
      typename Allocators::template shared_allocator_t<alignof(BufferElement)>;
  using AllocInboxOutbox = typename Allocators::template shared_allocator_t<64>;
  using AllocLocal = typename Allocators::template local_allocator_t<64>;
  using AllocRemote = typename Allocators::template remote_allocator_t<64>;

  using client_type = client<BufferElement, Word, SZ>;
  using server_type = server<BufferElement, Word, SZ>;

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

  HOSTRPC_ANNOTATE client_server_pair_t(SZ sz, ClientArch c, ServerArch s)
      : storage(host_client(
            Allocators(c, s).template shared_allocator<alignof(BufferElement)>(),
            Allocators(c, s).template shared_allocator<64>(),
            Allocators(c, s).template local_allocator<64>(),
            Allocators(c, s).template remote_allocator<64>(), sz, &server,
            &client))

  {
  }

  HOSTRPC_ANNOTATE ~client_server_pair_t() { storage.destroy(); }

  HOSTRPC_ANNOTATE bool valid() { return storage.valid(); }

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

template <typename SZ, typename ClientArch, typename ServerArch>
client_server_pair_t<SZ, ClientArch, ServerArch>
make_client_server_pair(SZ sz, ClientArch c, ServerArch s)
{
  return {sz, c, s};
}

}  // namespace hostrpc
#endif
