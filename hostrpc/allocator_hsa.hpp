#ifndef ALLOCATOR_HSA_HPP_INCLUDED
#define ALLOCATOR_HSA_HPP_INCLUDED

#include "allocator.hpp"

#include "detail/platform_detect.h"
#include <stdint.h>

namespace hostrpc
{
namespace allocator
{
namespace hsa_impl
{
void* allocate(uint64_t hsa_region_t_handle, size_t align, size_t bytes);
int deallocate(void*);
}  // namespace hsa_impl

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
    return {hsa_impl::allocate(hsa_region_t_handle, Align, N)};
  }
  static status destroy(raw x)
  {
    return (hsa_impl::deallocate(x.ptr) == 0) ? success : failure;
  }
};
}  // namespace allocator
}  // namespace hostrpc

#endif
