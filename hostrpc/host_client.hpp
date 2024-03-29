#ifndef HOSTRPC_HOST_CLIENT_HPP_INCLUDED
#define HOSTRPC_HOST_CLIENT_HPP_INCLUDED

#include "allocator.hpp"
#include "memory.hpp"

namespace hostrpc
{
template <typename T>
HOSTRPC_ANNOTATE size_t bytes_for_N_slots(size_t N)
{
  size_t bits = hostrpc::round8(N);
  return bits / 8;
}

template <typename T, typename U>
struct is_same_or_inverted
{
  static constexpr bool value =
    cxx::is_same<T, U>() || cxx::is_same<T, typename U::invertedType>();
};

// local, remote are instances of client_impl, server_impl
// essentially a misnamed constructor for allocator::store_impl

// may want to embed the architecture type in the client/server, kind of messy
// carrying it around like this
// allocators are stateful, and the state is not bound to the client/server
// instance,

template <typename SZ, typename LocalType, typename RemoteType,
          typename AllocBuffer, typename AllocInboxOutbox, typename AllocLocal,
          typename AllocRemote>
HOSTRPC_ANNOTATE allocator::store_impl<AllocBuffer, AllocInboxOutbox,
                                       AllocLocal, AllocRemote>
host_client(AllocBuffer alloc_buffer, AllocInboxOutbox alloc_inbox_outbox,
            AllocLocal alloc_local, AllocRemote alloc_remote, SZ sz,
            LocalType* local, RemoteType* remote)
{
  // consistency constraints (todo: let word vary across the two?)
  static_assert(
                cxx::is_same<typename LocalType::Word, typename RemoteType::Word>(), "");
  static_assert(cxx::is_same<typename LocalType::SZ, typename RemoteType::SZ>(),
                "");

  // can have different values for Inverted without problems
  // would like different types of mail box so that inbox doesn't have
  // write methods on it
#if 0
  static_assert(is_same_or_inverted<typename LocalType::inbox_t,
                typename RemoteType::outbox_t>(),
                "");

  static_assert(is_same_or_inverted<typename LocalType::outbox_t,
                typename RemoteType::inbox_t>(),
                "");
#endif
  
  static_assert(AllocBuffer::align == alignof(page_t), "");
  static_assert(AllocInboxOutbox::align == 64, "");

  // hazard - there is no requirement from the standard that memory which is
  // initially zero will remain so after placement new over it. Similarly,
  // should probably placement new into the remote memory, despite it being
  // inaccessible.

  using res_ty = allocator::store_impl<AllocBuffer, AllocInboxOutbox,
                                       AllocLocal, AllocRemote>;
#if (HOSTRPC_HOST)
  size_t N = sz.value();
  res_ty res = {
      alloc_buffer.allocate(sizeof(page_t) * N),
      alloc_inbox_outbox.allocate(
          bytes_for_N_slots<typename LocalType::inbox_t>(N)),
      alloc_inbox_outbox.allocate(
          bytes_for_N_slots<typename LocalType::outbox_t>(N)),
      alloc_local.allocate(bytes_for_N_slots<typename LocalType::lock_t>(N)),
      alloc_remote.allocate(bytes_for_N_slots<typename RemoteType::lock_t>(N))};

  // if any allocation failed, deallocate the others. may want to report
  if (!res.valid())
    {
      allocator::status rc = res.destroy();
      (void)rc;
      return res;
    }

  {
    auto recv = careful_cast_to_bitmap<typename LocalType::inbox_t>(
        res.recv.local_ptr(), N);
    auto send = careful_cast_to_bitmap<typename LocalType::outbox_t>(
        res.send.local_ptr(), N);

    auto* local_buffer =
        careful_array_cast<hostrpc::page_t>(res.buffer.local_ptr(), N);

    auto local_active = careful_cast_to_bitmap<typename LocalType::lock_t>(
        res.local_lock.local_ptr(), N);

    *local =
        (LocalType){sz, local_active, recv, send, local_buffer};
  }

  {
    // recv/send pointers swapped relative to local
    auto recv = careful_cast_to_bitmap<typename RemoteType::inbox_t>(
        res.send.remote_ptr(), N);
    auto send = careful_cast_to_bitmap<typename RemoteType::outbox_t>(
        res.recv.remote_ptr(), N);

    auto* remote_buffer =
        careful_array_cast<hostrpc::page_t>(res.buffer.remote_ptr(), N);

    auto remote_active = careful_cast_to_bitmap<typename RemoteType::lock_t>(
        res.remote_lock.remote_ptr(), N);

    *remote = (RemoteType){sz,   remote_active,  recv,
                           send, remote_buffer};
  }
#else
  // not yet implemented, need to do something with address space overloading
  (void)alloc_buffer;
  (void)alloc_inbox_outbox;
  (void)alloc_local;
  (void)alloc_remote;
  (void)sz;
  (void)local;
  (void)remote;
  res_ty res;
#endif
  return res;
}

}  // namespace hostrpc

#endif
