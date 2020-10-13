#ifndef HOSTRPC_MEMORY_CUDA_HPP_INCLUDED
#define HOSTRPC_MEMORY_CUDA_HPP_INCLUDED

#include <stddef.h>
#include <stdint.h>

namespace hostrpc
{
namespace cuda
{
// caller gets to align the result
// docs claim 'free' can return errors from unrelated launches (??), so I guess
// that should be propagated up
void *allocate_gpu(size_t size);
void deallocate_gpu(void *);

void *allocate_shared(size_t size);
void deallocate_shared(void *);

void *device_ptr_from_host_ptr(void *);

// derived from the above functions

// allocates and zero fills
void *allocate_gpu(size_t align, size_t size, void **to_free);
void *allocate_shared(size_t align, size_t size, void **to_free);

}  // namespace cuda
}  // namespace hostrpc

#endif
