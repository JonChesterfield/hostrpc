#ifndef HOSTRPC_HOST_CLIENT_HPP_INCLUDED
#define HOSTRPC_HOST_CLIENT_HPP_INCLUDED

#include "allocator.hpp"
#include <type_traits>

namespace hostrpc
{
template <typename T>
constexpr size_t bytes_for_N_slots(size_t N)
{
  constexpr size_t bps = T::bits_per_slot();
  static_assert(bps == 1 || bps == 8, "");
  constexpr size_t bits = round8(N * bps);
  return bits / 8;
}

// local, remote are instances of client_impl, server_impl
template <typename LocalType, typename RemoteType, typename AllocBuffer,
          typename AllocInboxOutbox, typename AllocLocal, typename AllocRemote>

struct host_client_t
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

  using storage_type =
      store_impl<AllocBuffer, AllocInboxOutbox, AllocLocal, AllocRemote>;

  storage_type host_client_t(AllocBuffer alloc_buffer,
                             AllocInboxOutbox alloc_inbox_outbox,
                             AllocLocal alloc_local, AllocRemote alloc_remote,
                             SZ sz, LocalType* local, RemoteType* remote)
  {
    // check SZ has same type as local/remote
    size_t N = sz.N();

    storage_type res = {
        AllocInboxOutbox.allocate(bytes_for_N_slots<local_type::inbox_t>(N)),
        AllocInboxOutbox.allocate(bytes_for_N_slots<local_type::outbox_t>(N)),
        AllocLocal.allocate(bytes_for_N_slots<local_type::lock_t>(N)),
        AllocLocal.allocate(bytes_for_N_slots<local_type::staging_t>(N)),
        AllocLocal.allocate(bytes_for_N_slots<remote_type::lock_t>(N)),
        AllocLocal.allocate(bytes_for_N_slots<remote_type::staging_t>(N))};

    // if any allocation failed, deallocate the others. may want to report
    if (!res.valid())
      {
        status rc = res.destroy();
        (void)rc;
        return res;
      }

    hostrpc::page_t* local_buffer =
        hostrpc::careful_array_cast<page_t>(res.buffer.local());
    hostrpc::page_t* remote_buffer =
        hostrpc::careful_array_cast<page_t>(res.buffer.remote());

    // write to local/remote pointer, should factor out the casts

    return res;
  }
};

}  // namespace hostrpc

#endif
