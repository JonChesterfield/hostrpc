#ifndef HOSTRPC_HOST_CLIENT_HPP_INCLUDED
#define HOSTRPC_HOST_CLIENT_HPP_INCLUDED

#include "allocator.hpp"
#include <type_traits>

namespace hostrpc
{
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
    auto page_buffer = alloc_buffer.allocate(N * sizeof(page_t));

    auto recv =
        AllocInboxOutbox.allocate(N * local_type::inbox_t::bits_per_slot());
    auto send =
        AllocInboxOutbox.allocate(N * local_type::outbox_t::bits_per_slot());

    auto local_lock =
        AllocLocal.allocate(N * local_type::lock_t::bits_per_slot());
    auto local_staging =
        AllocLocal.allocate(N * local_type::staging_t::bits_per_slot());

    auto remote_lock =
        AllocLocal.allocate(N * remote_type::lock_t::bits_per_slot());
    auto remote_staging =
        AllocLocal.allocate(N * remote_type::staging_t::bits_per_slot());

    hostrpc::page_t* local_buffer =
        hostrpc::careful_array_cast<page_t>(page_buffer.local());
    hostrpc::page_t* remote_buffer =
        hostrpc::careful_array_cast<page_t>(page_buffer.remote());

    // write to local/remote pointer, should factor out the casts

    storage_type res = {page_buffer,   recv,        send,          local_lock,
                        local_staging, remote_lock, remote_staging};

    return res;
  }
};

}  // namespace hostrpc

#endif
