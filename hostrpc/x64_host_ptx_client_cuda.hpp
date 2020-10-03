#ifndef X64_HOST_PTX_CLIENT_CUDA_HPP_INCLUDED
#define X64_HOST_PTX_CLIENT_CUDA_HPP_INCLUDED

#include <stddef.h>

namespace hostrpc
{
namespace cuda
{
// allocates and zero fills
// caller gets to align the result
// docs claim 'free' can return errors from unrelated launches (??), so I guess
// that should be propagated up
void *allocate_gpu(size_t size);
void deallocate_gpu(void *);

void *allocate_shared(size_t size);
void deallocate_shared(void *);

void *device_ptr_from_host_ptr(void *);
}  // namespace cuda
}  // namespace hostrpc

#endif
