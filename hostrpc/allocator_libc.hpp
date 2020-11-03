#ifndef ALLOCATOR_LIBC_HPP_INCLUDED
#define ALLOCATOR_LIBC_HPP_INCLUDED

#include "allocator.hpp"

#include "detail/platform_detect.h"

#if (HOSTRPC_HOST)

#include <stdlib.h>
#include <string.h>
#include <utility>

namespace hostrpc
{
namespace allocator
{
template <size_t Align>
struct host_libc : public interface<Align, host_libc<Align>>
{
  using typename interface<Align, host_libc<Align>>::raw;
  host_libc() {}
  raw allocate(size_t N)
  {
    void *ptr = aligned_alloc(Align, N);
    if (ptr)
      {
        memset(ptr, 0, N);
      }
    return {ptr};
  }
  static status destroy(raw x)
  {
    free(x.ptr);
    return success;
  }
};

}  // namespace allocator
}  // namespace hostrpc

#endif
#endif
