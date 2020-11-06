#ifndef ALLOCATOR_CUDA_HPP_INCLUDED
#define ALLOCATOR_CUDA_HPP_INCLUDED

#include "allocator.hpp"

#include "detail/platform_detect.h"

#include "memory_cuda.hpp"

namespace hostrpc
{
namespace allocator
{
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
    return {hostrpc::cuda::allocate_shared(adj)};  // zeros
  }
  static status destroy(raw x)
  {
    return hostrpc::cuda::deallocate_shared(x.ptr) == 0 ? success : failure;
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
    return {hostrpc::cuda::allocate_gpu(adj)};  // zeros
  }
  static status destroy(raw x)
  {
    return hostrpc::cuda::deallocate_gpu(x.ptr) == 0 ? success : failure;
  }
  static local_t local(raw)
  {
    // local is on the host (as only have a host cuda allocator at present)
    return {0};
  }
  static remote_t remote(raw x)
  {
    return {hostrpc::cuda::align_pointer_up(x.ptr, Align)};
  }
};

}  // namespace allocator
}  // namespace hostrpc

#endif
