#ifndef HOSTRPC_HOST_CLIENT_HPP_INCLUDED
#define HOSTRPC_HOST_CLIENT_HPP_INCLUDED

#include "allocator.hpp"
#include <type_traits>

namespace hostrpc
{
template <typename T>
size_t bytes_for_N_slots(size_t N)
{
  constexpr size_t bps = T::bits_per_slot();
  static_assert(bps == 1 || bps == 8, "");
  size_t bits = hostrpc::round8(N * bps);
  return bits / 8;
}

// local, remote are instances of client_impl, server_impl
template <typename SZ, typename LocalType, typename RemoteType,
          typename AllocBuffer, typename AllocInboxOutbox, typename AllocLocal,
          typename AllocRemote>
allocator::store_impl<AllocBuffer, AllocInboxOutbox, AllocLocal, AllocRemote>
host_client(AllocBuffer alloc_buffer, AllocInboxOutbox alloc_inbox_outbox,
            AllocLocal alloc_local, AllocRemote alloc_remote, SZ sz,
            LocalType* local, RemoteType* remote)
{
  // consistency constraints
  static_assert(
      std::is_same<typename LocalType::Word, typename RemoteType::Word>::value,
      "");
  static_assert(
      std::is_same<typename LocalType::SZ, typename RemoteType::SZ>::value, "");

  static_assert(std::is_same<typename LocalType::inbox_t,
                             typename RemoteType::outbox_t>::value,
                "");

  static_assert(std::is_same<typename LocalType::outbox_t,
                             typename RemoteType::inbox_t>::value,
                "");

  static_assert(AllocBuffer::align == alignof(page_t), "");
  static_assert(AllocInboxOutbox::align == 64, "");

  // check SZ has same type as local/remote
  size_t N = sz.N();

  // hazard - there is no requirement from the standard that memory which is
  // initially zero will remain to after placement new over it. Similarly,
  // should probably placement new into the remote memory, despite it being
  // inaccessible.

  allocator::store_impl<AllocBuffer, AllocInboxOutbox, AllocLocal, AllocRemote>
      res = {alloc_buffer.allocate(sizeof(page_t) * N),
             alloc_inbox_outbox.allocate(
                 bytes_for_N_slots<typename LocalType::inbox_t>(N)),
             alloc_inbox_outbox.allocate(
                 bytes_for_N_slots<typename LocalType::outbox_t>(N)),
             alloc_local.allocate(
                 bytes_for_N_slots<typename LocalType::lock_t>(N)),
             alloc_local.allocate(
                 bytes_for_N_slots<typename LocalType::staging_t>(N)),
             alloc_remote.allocate(
                 bytes_for_N_slots<typename RemoteType::lock_t>(N)),
             alloc_remote.allocate(
                 bytes_for_N_slots<typename RemoteType::staging_t>(N))};

  // if any allocation failed, deallocate the others. may want to report
  if (!res.valid())
    {
      allocator::status rc = res.destroy();
      (void)rc;
      return res;
    }

  {
    auto recv = careful_cast_to_bitmap<typename LocalType::inbox_t>(
        res.recv.local(), N);
    auto send = careful_cast_to_bitmap<typename LocalType::outbox_t>(
        res.send.local(), N);

    auto* local_buffer =
        careful_array_cast<hostrpc::page_t>(res.buffer.local(), N);

    auto local_active = careful_cast_to_bitmap<typename LocalType::lock_t>(
        res.local_lock.local(), N);
    auto local_staging = careful_cast_to_bitmap<typename LocalType::staging_t>(
        res.local_staging.local(), N);

    *local = (LocalType){sz,           local_active, recv, send, local_staging,
                         local_buffer, local_buffer};
  }

  {
    // recv/send pointers swapped relative to local
    auto recv = careful_cast_to_bitmap<typename RemoteType::inbox_t>(
        res.send.remote(), N);
    auto send = careful_cast_to_bitmap<typename RemoteType::outbox_t>(
        res.recv.remote(), N);

    auto* remote_buffer =
        careful_array_cast<hostrpc::page_t>(res.buffer.remote(), N);

    auto remote_active = careful_cast_to_bitmap<typename RemoteType::lock_t>(
        res.remote_lock.remote(), N);
    auto remote_staging =
        careful_cast_to_bitmap<typename RemoteType::staging_t>(
            res.remote_staging.remote(), N);

    *remote = (RemoteType){sz,           remote_active,  recv,
                           send,         remote_staging, remote_buffer,
                           remote_buffer};
  }

  return res;
}

}  // namespace hostrpc

#endif
