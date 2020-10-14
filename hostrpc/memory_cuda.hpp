#ifndef HOSTRPC_MEMORY_CUDA_HPP_INCLUDED
#define HOSTRPC_MEMORY_CUDA_HPP_INCLUDED

#include <assert.h>
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

inline void *align_pointer_up(void *ptr, size_t align)
{
  uint64_t top_misaligned = (uint64_t)ptr + align - 1;
  uint64_t aligned = top_misaligned & ~(align - 1);
  return (void *)aligned;
}

template <void *(*F)(size_t)>
inline void *aligned_allocate(size_t align, size_t size, void **to_free)
{
  assert(align != 0);
  assert((align & (align - 1)) == 0);  // align is power two
  void *base = F(size + align - 1);
  if (!base)
    {
      return base;
    }

  *to_free = base;
  return align_pointer_up(base, align);
}

// allocates and zero fills
inline void *allocate_gpu(size_t align, size_t size, void **to_free)
{
  return aligned_allocate<allocate_gpu>(align, size, to_free);
}

inline void *allocate_shared(size_t align, size_t size, void **to_free)
{
  return aligned_allocate<allocate_shared>(align, size, to_free);
}

}  // namespace cuda
}  // namespace hostrpc

#endif
