#ifndef HOSTRPC_MEMORY_HPP
#define HOSTRPC_MEMORY_HPP

#include <stddef.h>
#include <stdint.h>

namespace hostrpc
{
namespace x64_native
{
void* allocate(size_t align, size_t bytes);
void deallocate(void*);
}  // namespace x64_native

namespace hsa
{
void* allocate(uint64_t hsa_region_t_handle, size_t align, size_t bytes);
void deallocate(void*);
}  // namespace hsa
}  // namespace hostrpc

#endif
