#ifndef ALLOCATOR_HPP_INCLUDED
#define ALLOCATOR_HPP_INCLUDED

#include "detail/platform_detect.h"

#if (HOSTRPC_HOST)

#include <stdlib.h>
#include <utility>

#include "/home/amd/aomp/rocr-runtime/src/inc/hsa.h"
#include "/home/amd/aomp/rocr-runtime/src/inc/hsa_ext_amd.h"
#include "memory_cuda.hpp"

namespace hostrpc
{
namespace allocator
{
typedef enum
{
  success = 0,
  failure = 1,
} status;

struct res_t
{
  status res;
  void *local;
  void *remote;
  void *tofree;
};

template <typename Base>
struct scalar
{
  // local = nullptr on failure
  res_t allocate(size_t A, size_t N)
  {
    return static_cast<Base *>(this)->allocate_impl(A, N);
  }

  status deallocate(void *p)
  {
    return static_cast<Base *>(this)->deallocate_impl(p);
  }
};

// empty base doesn't work across multiple bases
template <typename Local, typename Shared, typename Remote>
struct composite
{
  Local local;
  Shared shared;
  Remote remote;

  composite(Local &&local, Shared &&shared, Remote &&remote)
      : local(local), shared(shared), remote(remote)
  {
  }

  res_t allocate_local(size_t A, size_t N) { return local.allocate(A, N); }
  status deallocate_local(void *p) { return local.deallocate(p); }

  res_t allocate_shared(size_t A, size_t N) { return shared.allocate(A, N); }
  status deallocate_shared(void *p) { return shared.deallocate(p); }

  res_t allocate_remote(size_t A, size_t N) { return remote.allocate(A, N); }
  status deallocate_remote(void *p) { return remote.deallocate(p); }
};

template <typename Local, typename Shared, typename Remote>
composite<Local, Shared, Remote> make_composite(Local &&local, Shared &&shared,
                                                Remote &&remote)
{
  return composite<Local, Shared, Remote>(std::move(local), std::move(shared),
                                          std::move(remote));
}

struct host_libc : public scalar<host_libc>
{
  res_t allocate_impl(size_t A, size_t N)
  {
    void *r = aligned_alloc(A, N);
    if (r)
      {
        return {success, r, r, r};
      }
    else
      {
        return {failure, nullptr, nullptr, nullptr};
      }
  }

  status deallocate_impl(void *ptr)
  {
    free(ptr);
    return success;
  }
};

struct hsa : public scalar<hsa>
{
  uint64_t hsa_region_t_handle;
  hsa(uint64_t hsa_region_t_handle) : hsa_region_t_handle(hsa_region_t_handle)
  {
  }

  res_t allocate_impl(size_t A, size_t N)
  {
    assert(A >= 64);
    assert(A <= 4096);  // todo

    (void)A;
    hsa_region_t region{.handle = hsa_region_t_handle};

    size_t bytes = 4 * ((N + 3) / 4);  // fill uses a multiple of four

    void *memory;
    if (HSA_STATUS_SUCCESS == hsa_memory_allocate(region, bytes, &memory))
      {
        // probably want memset for fine grain, may want it for gfx9
        // memset(memory, 0, bytes);
        // warning: This is likely to be relied on by bitmap
        hsa_status_t r = hsa_amd_memory_fill(memory, 0, bytes / 4);
        if (HSA_STATUS_SUCCESS == r)
          {
            return {
                success,
                memory,
                memory,
                memory,
            };
          }
      }

    return {failure, nullptr, nullptr, nullptr};
  }

  status deallocate_impl(void *ptr)
  {
    return (hsa_memory_free(ptr) == HSA_STATUS_SUCCESS) ? success : failure;
  }
};

namespace cuda
{
template <void *(*Alloc)(size_t, size_t, void **), void (*Free)(void *)>
res_t allocate(size_t A, size_t N)
{
  void *tofree;
  void *host = Alloc(A, N, &tofree);
  if (host)
    {
      void *device = hostrpc::cuda::device_ptr_from_host_ptr(host);
      if (device)
        {
          return {success, host, device, tofree};
        }
      else
        {
          // if deallocate fails here, can't do better elsewhere
          Free(tofree);
        }
    }
  return {failure, nullptr, nullptr, nullptr};
}

template <void (*Free)(void *)>
status deallocate(void *ptr)
{
  // todo: pass the error back
  Free(ptr);
  return success;
}
}  // namespace cuda

struct cuda_shared : scalar<cuda_shared>
{
  res_t allocate_impl(size_t A, size_t N)
  {
    return cuda::allocate<hostrpc::cuda::allocate_shared,
                          hostrpc::cuda::deallocate_shared>(A, N);
  }
  status deallocate_impl(void *ptr)
  {
    return cuda::deallocate<hostrpc::cuda::deallocate_shared>(ptr);
  }
};

struct cuda_gpu : scalar<cuda_gpu>
{
  res_t allocate_impl(size_t A, size_t N)
  {
    return cuda::allocate<hostrpc::cuda::allocate_gpu,
                          hostrpc::cuda::deallocate_gpu>(A, N);
  }
  status deallocate_impl(void *ptr)
  {
    return cuda::deallocate<hostrpc::cuda::deallocate_gpu>(ptr);
  }
};

// doesn't compile, classes need to be distinct
using all_libc = composite<host_libc, host_libc, host_libc>;

int use()
{
  host_libc hl;
  auto a = make_composite(std::move(hl), hsa(UINT64_C(42)), cuda_shared{});
  auto b = make_composite(cuda_gpu{}, cuda_shared{}, host_libc{});
  auto r = a.allocate_remote(64, 64);
  (void)a;
  (void)b;
  (void)r;
  return 0;
}

}  // namespace allocator

}  // namespace hostrpc

#endif  // (HOSTRPC_HOST)
#endif
