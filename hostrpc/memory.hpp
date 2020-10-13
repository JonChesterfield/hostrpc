#ifndef HOSTRPC_MEMORY_HPP_INCLUDED
#define HOSTRPC_MEMORY_HPP_INCLUDED

#include <stddef.h>
#include <stdint.h>

#if defined(__x86_64__)
#include "memory_hsa.hpp"
#include "memory_host.hpp"

#include <new>

namespace hostrpc
{
template <typename T>
T* careful_array_cast(void* data, size_t N)
{
  // allocation functions return void*, but casting that to a T* is not strictly
  // sufficient to satisfy the C++ object model. One should placement new
  // instead. Placement new on arrays isn't especially useful as it needs extra
  // space to store the size of the array, in order for delete[] to work.
  // Instead, walk the memory constructing each element individually.

  // Strictly one should probably do this with destructors as well. That seems
  // less necessary to avoid consequences from the aliasing rules.

  // Handles the invalid combination of nullptr data and N != 0 by returning the
  // cast nullptr, for convenience a the call site.
  T* typed = static_cast<T*>(data);
  if (data != nullptr)
    {
      for (size_t i = 0; i < N; i++)
        {
          new (typed + i) T;
        }
    }
  return typed;
}

template <typename T>
T careful_cast_to_bitmap(void* memory, size_t size)
{
  constexpr size_t bps = T::bits_per_slot();
  static_assert(bps == 1 || bps == 8, "");
  typename T::Ty* m =
      hostrpc::careful_array_cast<typename T::Ty>(memory, size * bps);
  return {m};
}

}

#endif

#endif
