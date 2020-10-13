#include "memory_hsa.hpp"

#include "hsa.h"
#include "hsa_ext_amd.h"

#include <assert.h>

namespace hostrpc
{
namespace hsa_amdgpu
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

void deallocate(void *d) { hsa_memory_free(d); }
}  // namespace hsa_amdgpu

}  // namespace hostrpc
