#include "memory_host.hpp"

#include <assert.h>
#include <stdlib.h>
#include <string.h>

namespace hostrpc
{
namespace x64_native
{
void *allocate(size_t align, size_t bytes)
{
  assert(align >= 64);
  void *memory = ::aligned_alloc(align, bytes);
  if (memory)
    {
      memset(memory, 0, bytes);
    }
  return memory;
}
void deallocate(void *d) { free(d); }
}  // namespace x64_native


}  // namespace hostrpc
