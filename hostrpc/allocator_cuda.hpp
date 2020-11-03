#ifndef ALLOCATOR_CUDA_HPP_INCLUDED
#define ALLOCATOR_CUDA_HPP_INCLUDED

#include "allocator.hpp"

#include "detail/platform_detect.h"

#if (HOSTRPC_HOST)

#include <stdlib.h>
#include <string.h>
#include <utility>

#include "memory_cuda.hpp"

namespace hostrpc
{
namespace allocator
{
template <bool Shared, size_t Align>
struct cuda_shared_gpu : public interface<Align, cuda_shared_gpu<Shared, Align>>
{
  using Base = interface<Align, cuda_shared_gpu<Shared, Align>>;
  using typename Base::local_t;
  using typename Base::raw;
  using typename Base::remote_t;

  cuda_shared_gpu() {}
  raw allocate(size_t N)
  {
    size_t adj = N + Align - 1;
    return {hostrpc::cuda::dispatch<Shared>::allocate(adj)};  // zeros
  }
  static status destroy(raw x)
  {
    return hostrpc::cuda::dispatch<Shared>::deallocate(x.ptr) == 0 ? success
                                                                   : failure;
  }
  static local_t local(raw x)
  {
    return {hostrpc::cuda::align_pointer_up(x.ptr, Align)};
  }
  static remote_t remote(raw x)
  {
    if (!x.ptr)
      {
        return {0};
      }
    void *dev = hostrpc::cuda::device_ptr_from_host_ptr(x.ptr);
    return {hostrpc::cuda::align_pointer_up(dev, Align)};
  }
};

template <size_t Align>
using cuda_shared = cuda_shared_gpu<true, Align>;

template <size_t Align>
using cuda_gpu = cuda_shared_gpu<false, Align>;

}  // namespace allocator
}  // namespace hostrpc

#endif
#endif
