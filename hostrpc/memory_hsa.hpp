#ifndef HOSTRPC_MEMORY_HSA_HPP_INCLUDED
#define HOSTRPC_MEMORY_HSA_HPP_INCLUDED

#include <stddef.h>
#include <stdint.h>

namespace hostrpc
{
namespace hsa_amdgpu
{
void* allocate(uint64_t hsa_region_t_handle, size_t align, size_t bytes);
int deallocate(void*);
}  // namespace hsa_amdgpu
}  // namespace hostrpc

#endif
