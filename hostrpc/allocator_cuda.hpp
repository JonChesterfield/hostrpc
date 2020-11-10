#ifndef ALLOCATOR_CUDA_HPP_INCLUDED
#define ALLOCATOR_CUDA_HPP_INCLUDED

#include "allocator.hpp"

#include "detail/platform_detect.h"

#include <stddef.h>
#include <stdint.h>

namespace hostrpc
{
namespace allocator
{
namespace cuda_impl
{
// caller gets to align the result
// docs claim 'free' can return errors from unrelated launches (??), so I guess
// that should be propagated up
void *allocate_gpu(size_t size);
int deallocate_gpu(void *);

void *allocate_shared(size_t size);
int deallocate_shared(void *);

void *device_ptr_from_host_ptr(void *);

inline void *align_pointer_up(void *ptr, size_t align)
{
  uint64_t top_misaligned = (uint64_t)ptr + align - 1;
  uint64_t aligned = top_misaligned & ~(align - 1);
  return (void *)aligned;
}

}  // namespace cuda_impl

template <size_t Align>
struct cuda_shared : public interface<Align, cuda_shared<Align>>
{
  // allocate a pointer on the host and derive a device one from it
  using Base = interface<Align, cuda_shared<Align>>;
  using typename Base::local_t;
  using typename Base::raw;
  using typename Base::remote_t;

  cuda_shared() {}
  raw allocate(size_t N)
  {
    size_t adj = N + Align - 1;
    return {cuda_impl::allocate_shared(adj)};  // zeros
  }
  static status destroy(raw x)
  {
    return cuda_impl::deallocate_shared(x.ptr) == 0 ? success : failure;
  }
  static local_t local(raw x)
  {
    return {cuda_impl::align_pointer_up(x.ptr, Align)};
  }
  static remote_t remote(raw x)
  {
    if (!x.ptr)
      {
        return {0};
      }
    void *dev = cuda_impl::device_ptr_from_host_ptr(x.ptr);
    return {cuda_impl::align_pointer_up(dev, Align)};
  }
};

template <size_t Align>
struct cuda_gpu : public interface<Align, cuda_gpu<Align>>
{
  using Base = interface<Align, cuda_gpu<Align>>;
  using typename Base::local_t;
  using typename Base::raw;
  using typename Base::remote_t;

  cuda_gpu() {}
  raw allocate(size_t N)
  {
    size_t adj = N + Align - 1;
    return {cuda_impl::allocate_gpu(adj)};  // zeros
  }
  static status destroy(raw x)
  {
    return cuda_impl::deallocate_gpu(x.ptr) == 0 ? success : failure;
  }
  static local_t local(raw)
  {
    // local is on the host (as only have a host cuda allocator at present)
    return {0};
  }
  static remote_t remote(raw x)
  {
    return {cuda_impl::align_pointer_up(x.ptr, Align)};
  }
};

}  // namespace allocator
}  // namespace hostrpc

#endif
