#ifndef TEST_COMMON_HPP_INCLUDED
#define TEST_COMMON_HPP_INCLUDED

// functions that are useful for multiple tests but not necessarily worth
// including in the library
#include "memory.hpp"

#if defined(__x86_64__)
#include "hsa.h"
#endif

namespace hostrpc
{
static constexpr size_t round(size_t x) { return 64u * ((x + 63u) / 64u); }
_Static_assert(0 == round(0), "");
_Static_assert(64 == round(1), "");
_Static_assert(64 == round(2), "");
_Static_assert(64 == round(63), "");
_Static_assert(64 == round(64), "");
_Static_assert(128 == round(65), "");
_Static_assert(128 == round(127), "");
_Static_assert(128 == round(128), "");
_Static_assert(192 == round(129), "");

#if defined(__x86_64__)
namespace
{
template <typename T>
T hsa_allocate_slot_bitmap_data_alloc(hsa_region_t region, size_t size)
{
  constexpr size_t bps = T::bits_per_slot();
  static_assert(bps == 1 || bps == 8, "");
  const size_t align = 64;
  void *memory =
      hostrpc::hsa_amdgpu::allocate(region.handle, align, size * bps);
  _Atomic(uint64_t) *m =
      hostrpc::careful_array_cast<_Atomic(uint64_t)>(memory, size * bps);
  return {m};
}

inline void hsa_allocate_slot_bitmap_data_free(_Atomic(uint64_t) * d)
{
  hostrpc::hsa_amdgpu::deallocate(static_cast<void *>(d));
}
}  // namespace
#endif
}  // namespace hostrpc

#endif
