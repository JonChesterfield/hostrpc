#include "allocator.hpp"

#include "detail/platform/detect.hpp"
#if !HOSTRPC_HOST
#error "allocator_host_libc relies on the host libc library"
#endif

#include <stdlib.h>
#include <string.h>

namespace hostrpc
{
namespace allocator
{
namespace host_libc_impl
{
void *allocate(size_t Align, size_t bytes)
{
  void *ptr = aligned_alloc(Align, bytes);
  if (ptr)
    {
      memset(ptr, 0, bytes);
    }
  return ptr;
}
void deallocate(void *ptr) { free(ptr); }
}  // namespace host_libc_impl

}  // namespace allocator
}  // namespace hostrpc
