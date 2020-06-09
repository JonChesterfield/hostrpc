#ifndef HOSTRPC_X64_HOST_AMDGCN_CLIENT_HPP_INCLUDED
#define HOSTRPC_X64_HOST_AMDGCN_CLIENT_HPP_INCLUDED

#include "client.hpp"
#include "platform.hpp"
#include "server.hpp"

#include "x64_host_amdgcn_client_api.hpp"

// hsa uses freestanding C headers, unlike hsa.hpp
#if !defined(__AMDGCN__)
#include "hsa.h"
#include <string.h>
#endif

namespace hostrpc
{
template <size_t size>
struct hsa_allocate_slot_bitmap_data
{
  constexpr const static size_t align = 64;
  static_assert(size % 64 == 0, "Size must be multiple of 64");

  alignas(align) _Atomic uint64_t data[size / 64];
};

namespace config
{
struct fill
{
  static void call(hostrpc::page_t *page, void *dv)
  {
    uint64_t *d = static_cast<uint64_t *>(dv);
    if (platform::is_master_lane())
      {
        for (unsigned i = 0; i < 64; i++)
          {
            page->cacheline[i].element[0] = 0;
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
  static void call(hostrpc::page_t *, void *){};
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
        printf("[%u:] %lu %lu %lu %lu %lu %lu %lu %lu\n", c, line.element[0],
               line.element[1], line.element[2], line.element[3],
               line.element[4], line.element[5], line.element[6],
               line.element[7]);
      }
  }
};
#endif

template <size_t N>
class x64_amdgcn_bitmap_types
{
 public:
  using inbox_t = slot_bitmap<N, __OPENCL_MEMORY_SCOPE_ALL_SVM_DEVICES,
                              hsa_allocate_slot_bitmap_data>;
  using outbox_t = slot_bitmap<N, __OPENCL_MEMORY_SCOPE_ALL_SVM_DEVICES,
                               hsa_allocate_slot_bitmap_data>;
  using locks_t = slot_bitmap<N, __OPENCL_MEMORY_SCOPE_DEVICE,
                              hsa_allocate_slot_bitmap_data>;
};

}  // namespace config

// need to allocate buffers for both together
// allocation functions are only available in the host

template <size_t N>
using x64_amdgcn_client =
    hostrpc::client<N, config::x64_amdgcn_bitmap_types,
                    hostrpc::copy_functor_memcpy_pull, config::fill,
                    config::use, hostrpc::nop_stepper>;

#if !defined(__AMDGCN__)
template <size_t N>
using x64_amdgcn_server =
    hostrpc::server<N, config::x64_amdgcn_bitmap_types,
                    hostrpc::copy_functor_memcpy_pull, config::operate,
                    hostrpc::nop_stepper>;
#endif

// TODO: Put this in an interface header
static const constexpr size_t x64_host_amdgcn_array_size =
    2048;  // needs to scale with CUs

#if !defined(__AMDGCN__)
namespace
{
// allocate/free are out of line as they only exist on the host
template <size_t size>
inline hsa_allocate_slot_bitmap_data<size> *hsa_allocate_slot_bitmap_data_alloc(
    hsa_region_t region)
{
  void *memory;
  hsa_status_t r = hsa_memory_allocate(region, size, &memory);
  if (r != HSA_STATUS_SUCCESS)
    {
      return nullptr;
    }

  return new (memory) hsa_allocate_slot_bitmap_data<size>;
}

template <size_t size>
inline void hsa_allocate_slot_bitmap_data_free(
    hsa_allocate_slot_bitmap_data<size> *d)
{
  hsa_memory_free(static_cast<void *>(d));
}

inline void *alloc_from_region(hsa_region_t region, size_t size)
{
  void *res;
  hsa_status_t r = hsa_memory_allocate(region, size, &res);
  return (r == HSA_STATUS_SUCCESS) ? res : nullptr;
}
}  // namespace

template <size_t N>
struct x64_amdgcn_pair
{
  using mt = slot_bitmap<N, __OPENCL_MEMORY_SCOPE_ALL_SVM_DEVICES,
                         hsa_allocate_slot_bitmap_data>;

  using lt = slot_bitmap<N, __OPENCL_MEMORY_SCOPE_DEVICE,
                         hsa_allocate_slot_bitmap_data>;

  x64_amdgcn_pair(hsa_region_t fine)
  {
    hostrpc::page_t *client_buffer =
        reinterpret_cast<page_t *>(alloc_from_region(fine, N * sizeof(page_t)));
    hostrpc::page_t *server_buffer =
        reinterpret_cast<page_t *>(alloc_from_region(fine, N * sizeof(page_t)));

    typename mt::slot_bitmap_data_t *send_data =
        hsa_allocate_slot_bitmap_data_alloc<N>(fine);
    typename mt::slot_bitmap_data_t *recv_data =
        hsa_allocate_slot_bitmap_data_alloc<N>(fine);

    typename lt::slot_bitmap_data_t *client_active_data =
        hsa_allocate_slot_bitmap_data_alloc<N>(fine);

    typename lt::slot_bitmap_data_t *server_active_data =
        hsa_allocate_slot_bitmap_data_alloc<N>(fine);

    mt send = (send_data);
    mt recv = (recv_data);
    lt client_active = (client_active_data);
    lt server_active = (server_active_data);

    client = {recv, send, client_active, server_buffer, client_buffer};

    server = {send, recv, server_active, client_buffer, server_buffer};

    {
      uint64_t data[client.serialize_size()];
      client.serialize(data);

      // sanity check deserialize
      x64_amdgcn_client<N> chk;
      chk.deserialize(data);
      assert(memcmp(&client, &chk, 40) == 0);
    }
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

    hsa_memory_free(client.local_buffer);
    hsa_memory_free(server.local_buffer);
  }

  x64_amdgcn_client<N> client;
  x64_amdgcn_server<N> server;
};
#endif
}  // namespace hostrpc

#endif
