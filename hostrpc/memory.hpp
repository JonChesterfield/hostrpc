#ifndef HOSTRPC_MEMORY_HPP_INCLUDED
#define HOSTRPC_MEMORY_HPP_INCLUDED

#include "base_types.hpp"
#include <stddef.h>
#include <stdint.h>

// TODO: Put these somewhere else  or include <new> and restrict to HOST
inline void* operator new(size_t, _Atomic(uint32_t)* p) { return p; }
inline void* operator new(size_t, _Atomic(uint64_t)* p) { return p; }
inline void* operator new(size_t, hostrpc::page_t* p) { return p; }

namespace hostrpc
{
// likely to need address_space(1) overloads. Can't placement new into it, so
// will be reinterpret_cast.

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
  typename T::Ty* m = hostrpc::careful_array_cast<typename T::Ty>(memory, size);
  return {m};
}

}  // namespace hostrpc

#endif
