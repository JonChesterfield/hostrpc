#ifndef ALLOCATOR_HSA_HPP_INCLUDED
#define ALLOCATOR_HSA_HPP_INCLUDED

#include "allocator.hpp"

#include "detail/platform_detect.h"

#include "memory_hsa.hpp"

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
    return {hostrpc::hsa_amdgpu::allocate(hsa_region_t_handle, Align, N)};
  }
  static status destroy(raw x)
  {
    return (hostrpc::hsa_amdgpu::deallocate(x.ptr) == 0) ? success : failure;
  }
};
}  // namespace allocator
}  // namespace hostrpc

#endif
