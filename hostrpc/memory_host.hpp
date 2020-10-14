#ifndef HOSTRPC_MEMORY_HOST_HPP_INCLUDED
#define HOSTRPC_MEMORY_HOST_HPP_INCLUDED

#include <stddef.h>
#include <stdint.h>

namespace hostrpc
{
namespace x64_native
{
void* allocate(size_t align, size_t bytes);
void deallocate(void*);
}  // namespace x64_native

}  // namespace hostrpc

#endif
