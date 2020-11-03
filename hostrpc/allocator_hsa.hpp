#ifndef ALLOCATOR_HSA_HPP_INCLUDED
#define ALLOCATOR_HSA_HPP_INCLUDED

#include "allocator.hpp"

#include "detail/platform_detect.h"

#include <utility>

#include "/home/amd/aomp/rocr-runtime/src/inc/hsa.h"
#include "/home/amd/aomp/rocr-runtime/src/inc/hsa_ext_amd.h"

#if (HOSTRPC_HOST)

namespace hostrpc
{
namespace allocator
{
template <size_t Align>
struct hsa : public interface<Align, hsa<Align>>
{
  using typename interface<Align, hsa<Align>>::raw;
  uint64_t hsa_region_t_handle;
  hsa(uint64_t hsa_region_t_handle) : hsa_region_t_handle(hsa_region_t_handle)
  {
  }
  raw allocate(size_t N)
  {
    hsa_region_t region{.handle = hsa_region_t_handle};

    size_t bytes = 4 * ((N + 3) / 4);  // fill uses a multiple of four

    void *memory;
    if (HSA_STATUS_SUCCESS == hsa_memory_allocate(region, bytes, &memory))
      {
        hsa_status_t r = hsa_amd_memory_fill(memory, 0, bytes / 4);
        if (HSA_STATUS_SUCCESS == r)
          {
            return {memory};
          }
      }

    return {nullptr};
  }
  static status destroy(raw x)
  {
    return (hsa_memory_free(x.ptr) == HSA_STATUS_SUCCESS) ? success : failure;
  }
};
}  // namespace allocator
}  // namespace hostrpc

#endif
#endif
