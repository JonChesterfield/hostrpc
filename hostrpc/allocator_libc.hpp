#ifndef ALLOCATOR_LIBC_HPP_INCLUDED
#define ALLOCATOR_LIBC_HPP_INCLUDED

#include "allocator.hpp"

#include "detail/platform_detect.h"

#if (HOSTRPC_HOST)
#include <stdlib.h>
#include <string.h>
#include <utility>
#endif

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
#if (HOSTRPC_HOST)
    void *ptr = aligned_alloc(Align, N);
    if (ptr)
      {
        memset(ptr, 0, N);
      }
    return {ptr};
#else
    (void)N;
    return {nullptr};
#endif
  }
  static status destroy(raw x)
  {
    (void)x;
#if (HOSTRPC_HOST)
    free(x.ptr);
#endif
    return success;
  }
};

}  // namespace allocator
}  // namespace hostrpc

#endif
