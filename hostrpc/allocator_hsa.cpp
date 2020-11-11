#include "allocator.hpp"

#include "hsa.h"
#include "hsa_ext_amd.h"
#include <assert.h>

#include "detail/platform_detect.h"
#if !HOSTRPC_HOST
#error "allocator_hsa relies on the hsa host library"
#endif

namespace hostrpc
{
namespace allocator
{
namespace hsa_impl
{
void *allocate(uint64_t hsa_region_t_handle, size_t align, size_t bytes)
{
  assert(align >= 64);
  (void)align;  // todo
  hsa_region_t region{.handle = hsa_region_t_handle};

  bytes = 4 * ((bytes + 3) / 4);  // fill uses a multiple of four

  void *memory;
  if (HSA_STATUS_SUCCESS == hsa_memory_allocate(region, bytes, &memory))
    {
      // probably want memset for fine grain, may want it for gfx9
      // memset(memory, 0, bytes);
      // warning: This is likely to be relied on by bitmap
      hsa_status_t r = hsa_amd_memory_fill(memory, 0, bytes / 4);
      if (HSA_STATUS_SUCCESS == r)
        {
          return memory;
        }
    }

  return nullptr;
}

int deallocate(void *d)
{
  return (hsa_memory_free(d) == HSA_STATUS_SUCCESS) ? 0 : 1;
}

}  // namespace hsa_impl

}  // namespace allocator
}  // namespace hostrpc
