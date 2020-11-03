#ifndef ALLOCATOR_HPP_INCLUDED
#define ALLOCATOR_HPP_INCLUDED

#include "detail/platform_detect.h"

#include <stddef.h>

namespace hostrpc
{
namespace allocator
{
typedef enum
{
  success = 0,
  failure = 1,
} status;

template <size_t Align, typename Base>
struct interface
{
  _Static_assert(Align != 0, "");
  _Static_assert((Align & (Align - 1)) == 0, "");  // align is power two

  // some data is cache line aligned. hsa page aligns by default.
  _Static_assert((Align >= 64) && (Align <= 4096), "Current assumption");

  struct local_t
  {
    local_t(void *p) : ptr(p) {}
    void *ptr;
  };

  struct remote_t
  {
    remote_t(void *p) : ptr(p) {}
    void *ptr;
  };

  struct raw;
  raw allocate(size_t A, size_t N)
  {
    return static_cast<Base *>(this)->allocate(A, N);
  }

  struct raw
  {
    raw(void *p) : ptr(p) {}
    int destroy() { return Base::destroy(*this); }
    local_t local() { return Base::local(*this); }
    remote_t remote() { return Base::remote(*this); }
    void *ptr;
  };

 private:
  // The raw/local/remote conversion is a no-op for most allocators
  static local_t local(raw x) { return x.ptr; }
  static remote_t remote(raw x) { return x.ptr; }
};

}  // namespace allocator
}  // namespace hostrpc

#if (HOSTRPC_HOST)

#include <stdlib.h>
#include <string.h>
#include <utility>

#include "/home/amd/aomp/rocr-runtime/src/inc/hsa.h"
#include "/home/amd/aomp/rocr-runtime/src/inc/hsa_ext_amd.h"
#include "memory_cuda.hpp"

namespace hostrpc
{
namespace allocator
{
template <size_t Align>
struct hsa : public interface<Align, hsa<Align>>
{
  using typename interface<Align, hsa<Align>>::raw;
  uint64_t hsa_region_t_handle;
  hsa(uint64_t hsa_region_t_handle) : hsa_region_t_handle(hsa_region_t_handle)
  {
  }
  raw allocate(size_t N)
  {
    hsa_region_t region{.handle = hsa_region_t_handle};

    size_t bytes = 4 * ((N + 3) / 4);  // fill uses a multiple of four

    void *memory;
    if (HSA_STATUS_SUCCESS == hsa_memory_allocate(region, bytes, &memory))
      {
        hsa_status_t r = hsa_amd_memory_fill(memory, 0, bytes / 4);
        if (HSA_STATUS_SUCCESS == r)
          {
            return {memory};
          }
      }

    return {nullptr};
  }
  static status destroy(raw x)
  {
    return (hsa_memory_free(x.ptr) == HSA_STATUS_SUCCESS) ? success : failure;
  }
};

template <size_t Align>
struct host_libc : public interface<Align, host_libc<Align>>
{
  using typename interface<Align, host_libc<Align>>::raw;
  host_libc() {}
  raw allocate(size_t N)
  {
    void *ptr = aligned_alloc(Align, N);
    if (ptr)
      {
        memset(ptr, 0, N);
      }
    return {ptr};
  }
  static status destroy(raw x)
  {
    free(x.ptr);
    return success;
  }
};

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

#endif  // (HOSTRPC_HOST)
#endif
