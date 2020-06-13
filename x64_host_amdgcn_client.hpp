#ifndef HOSTRPC_X64_HOST_AMDGCN_CLIENT_HPP_INCLUDED
#define HOSTRPC_X64_HOST_AMDGCN_CLIENT_HPP_INCLUDED

#include "client_impl.hpp"
#include "platform.hpp"
#include "server_impl.hpp"

#include "x64_host_amdgcn_client_api.hpp"

// hsa uses freestanding C headers, unlike hsa.hpp
#if !defined(__AMDGCN__)
#include "hsa.h"
#include <string.h>
#endif

namespace hostrpc
{
namespace x64_host_amdgcn_client
{
struct fill
{
  static void call(hostrpc::page_t *page, void *dv)
  {
    uint64_t *d = static_cast<uint64_t *>(dv);
    if (0)
      {
        // Will want to set inactive lanes to nop here, once there are some
        if (platform::is_master_lane())
          {
            for (unsigned i = 0; i < 64; i++)
              {
                page->cacheline[i].element[0] = 0;
              }
          }
      }

    hostrpc::cacheline_t *line = &page->cacheline[platform::get_lane_id()];
    for (unsigned i = 0; i < 8; i++)
      {
        line->element[i] = d[i];
      }
  };
};

struct use
{
  static void call(hostrpc::page_t *page, void *dv)
  {
    uint64_t *d = static_cast<uint64_t *>(dv);
    hostrpc::cacheline_t *line = &page->cacheline[platform::get_lane_id()];
    for (unsigned i = 0; i < 8; i++)
      {
        d[i] = line->element[i];
      }
  };
};

#if !defined(__AMDGCN__)
// printf isn't available on target yet
struct operate
{
  static void call(hostrpc::page_t *page, void *)
  {
    for (unsigned c = 0; c < 64; c++)
      {
        hostrpc::cacheline_t &line = page->cacheline[c];
#if 1
        for (unsigned i = 0; i < 8; i++)
          {
            line.element[i] = 2 * (line.element[i] + 1);
          }
#else
        for (unsigned e = 0; e < 8; e++)
          {
            uint64_t elt = line.element[e];
            line.element[e] = elt * elt;
          }
#endif
      }
  }
};
#endif

}  // namespace x64_host_amdgcn_client

// need to allocate buffers for both together
// allocation functions are only available in the host

template <typename SZ>
using x64_amdgcn_client =
    hostrpc::client_impl<SZ, hostrpc::copy_functor_given_alias,
                         x64_host_amdgcn_client::fill,
                         x64_host_amdgcn_client::use, hostrpc::nop_stepper>;

#if !defined(__AMDGCN__)
template <typename SZ>
using x64_amdgcn_server =
    hostrpc::server_impl<SZ, hostrpc::copy_functor_given_alias,
                         x64_host_amdgcn_client::operate, hostrpc::nop_stepper>;
#endif

// needs to scale with CUs
static const constexpr size_t x64_host_amdgcn_array_size = 2048;

#if !defined(__AMDGCN__)
namespace
{
template <size_t size>
inline _Atomic uint64_t *hsa_allocate_slot_bitmap_data_alloc(
    hsa_region_t region)
{
  const size_t align = 64;
  void *memory = hostrpc::hsa::allocate(region.handle, align, size);
  return reinterpret_cast<_Atomic uint64_t *>(memory);
}

template <size_t size>
inline void hsa_allocate_slot_bitmap_data_free(_Atomic uint64_t *d)
{
  hostrpc::hsa::deallocate(static_cast<void *>(d));
}

inline void *alloc_from_region(hsa_region_t region, size_t size)
{
  return hostrpc::hsa::allocate(region.handle, 8, size);
}
}  // namespace

template <size_t N>
struct x64_amdgcn_pair
{
  using SZ = hostrpc::size_compiletime<N>;
  x64_amdgcn_client<SZ> client;
  x64_amdgcn_server<SZ> server;

  x64_amdgcn_pair(hsa_region_t fine, hsa_region_t gpu_coarse)
  {
    // todo: alignment on the page_t, works at present because allocate has high
    // granularity
    hostrpc::page_t *client_buffer =
        reinterpret_cast<page_t *>(alloc_from_region(fine, N * sizeof(page_t)));
    hostrpc::page_t *server_buffer = client_buffer;

    auto *send_data = hsa_allocate_slot_bitmap_data_alloc<N>(fine);
    auto *recv_data = hsa_allocate_slot_bitmap_data_alloc<N>(fine);
    auto *client_active_data =
        hsa_allocate_slot_bitmap_data_alloc<N>(gpu_coarse);
    auto *server_active_data = hsa_allocate_slot_bitmap_data_alloc<N>(fine);

    const size_t size = N;
    slot_bitmap_all_svm send = {size, send_data};
    slot_bitmap_all_svm recv = {size, recv_data};
    slot_bitmap_device client_active = {size, client_active_data};
    slot_bitmap_device server_active = {size, server_active_data};

    client = {SZ{}, recv, send, client_active, server_buffer, client_buffer};

    server = {SZ{}, send, recv, server_active, client_buffer, server_buffer};
  }

  ~x64_amdgcn_pair()
  {
    assert(client.inbox.data() == server.outbox.data());
    assert(client.outbox.data() == server.inbox.data());

    hsa_allocate_slot_bitmap_data_free<N>(client.inbox.data());
    hsa_allocate_slot_bitmap_data_free<N>(client.outbox.data());

    hsa_allocate_slot_bitmap_data_free<N>(client.active.data());
    hsa_allocate_slot_bitmap_data_free<N>(server.active.data());

    assert(client.local_buffer == server.remote_buffer);
    assert(client.remote_buffer == server.local_buffer);

    if (client.local_buffer == client.remote_buffer)
      {
        hsa_memory_free(client.local_buffer);
      }
    else
      {
        hsa_memory_free(client.local_buffer);
        hsa_memory_free(server.local_buffer);
      }
  }
};
#endif
}  // namespace hostrpc

#endif
