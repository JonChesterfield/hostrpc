#ifndef HOSTRPC_X64_HOST_AMDGCN_CLIENT_HPP_INCLUDED
#define HOSTRPC_X64_HOST_AMDGCN_CLIENT_HPP_INCLUDED

#include "client.hpp"
#include "server.hpp"

// hsa uses freestanding C headers, unlike hsa.hpp
#include "hsa.h"

namespace hostrpc
{
template <size_t size>
struct hsa_allocate_slot_bitmap_data
{
  constexpr const static size_t align = 64;
  static_assert(size % 64 == 0, "Size must be multiple of 64");

  static hsa_allocate_slot_bitmap_data *alloc(hsa_region_t region)
  {
    void *memory;
    hsa_status_t r = hsa_memory_allocate(region, size, &memory);
    if (r != HSA_STATUS_SUCCESS)
      {
        return nullptr;
      }

    return new (memory) hsa_allocate_slot_bitmap_data;
  }
  static void free(hsa_allocate_slot_bitmap_data *d) { (void)d; }
  alignas(align) _Atomic uint64_t data[size / 64];

  struct deleter
  {
    void operator()(hsa_allocate_slot_bitmap_data *d)
    {
      hsa_allocate_slot_bitmap_data::free(d);
    }
  };

 private:
  hsa_region_t region;
};

namespace config
{
struct fill
{
  void operator()(hostrpc::page_t *page, void *dv)
  {
    __builtin_memcpy(page, dv, sizeof(hostrpc::page_t));
  };
};

struct use
{
  void operator()(hostrpc::page_t *page, void *dv)
  {
    __builtin_memcpy(dv, page, sizeof(hostrpc::page_t));
  };
};

struct operate
{
  void operator()(hostrpc::page_t *page, void *)
  {
    for (unsigned c = 0; c < 64; c++)
      {
        hostrpc::cacheline_t &line = page->cacheline[c];
        std::swap(line.element[0], line.element[7]);
        std::swap(line.element[1], line.element[6]);
        std::swap(line.element[2], line.element[5]);
        std::swap(line.element[3], line.element[4]);
        for (unsigned i = 0; i < 8; i++)
          {
            line.element[i]++;
          }
      }
  }
};

// 128 won't suffice, probably need the whole structure to be templated
using x64_amdgcn_client =
    hostrpc::client<128, hostrpc::hsa_allocate_slot_bitmap_data,
                    hostrpc::copy_functor_memcpy_pull, fill, use,
                    hostrpc::nop_stepper>;

using x64_amdgcn_server =
    hostrpc::server<128, hostrpc::hsa_allocate_slot_bitmap_data,
                    hostrpc::copy_functor_memcpy_pull, operate,
                    hostrpc::nop_stepper>;

}  // namespace config

// need to allocate buffers for both together
// one needs to reside on the host, one on the gpu
struct x64_amdgcn_pair
{
  x64_amdgcn_pair();
  ~x64_amdgcn_pair();
  config::x64_amdgcn_client *client;
  config::x64_amdgcn_server *server;
  void *state;
};

}  // namespace hostrpc

#endif
