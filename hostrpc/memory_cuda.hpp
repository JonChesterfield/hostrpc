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
int deallocate_gpu(void *);

void *allocate_shared(size_t size);
int deallocate_shared(void *);

template <bool Shared>
struct dispatch;

template <>
struct dispatch<true>
{
  static void *allocate(size_t size) { return allocate_shared(size); }
  static int deallocate(void *ptr) { return deallocate_shared(ptr); }
};

template <>
struct dispatch<false>
{
  static void *allocate(size_t size) { return allocate_gpu(size); }
  static int deallocate(void *ptr) { return deallocate_gpu(ptr); }
};

void *device_ptr_from_host_ptr(void *);

inline void *align_pointer_up(void *ptr, size_t align)
{
  uint64_t top_misaligned = (uint64_t)ptr + align - 1;
  uint64_t aligned = top_misaligned & ~(align - 1);
  return (void *)aligned;
}

}  // namespace cuda
}  // namespace hostrpc

#endif
