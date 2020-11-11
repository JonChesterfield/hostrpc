#include "allocator.hpp"

#include <cuda.h>
#include <cuda_runtime.h>

#include "detail/platform_detect.h"
#if !HOSTRPC_HOST
#error "allocator_cuda relies on the cuda host library"
#endif

namespace hostrpc
{
namespace allocator
{
namespace cuda_impl
{
void *allocate_gpu(size_t size)
{
  void *ptr;
  cudaError_t rc = cudaMalloc(&ptr, size);
  if (rc != cudaSuccess)
    {
      return nullptr;
    }

  // this runs asychronously with the host
  // documentation is not clear on how to tell when it has finished
  rc = cudaMemset(ptr, 0, size);
  if (rc != cudaSuccess)
    {
      return nullptr;
    }

  rc = cudaDeviceSynchronize();
  if (rc != cudaSuccess)
    {
      return nullptr;
    }

  return ptr;
}

int deallocate_gpu(void *ptr)
{
  cudaError_t rc = cudaFree(ptr);
  return rc == cudaSuccess ? 0 : 1;
}

void *allocate_shared(size_t size)
{
  // cudaHostRegister may be a better choice as the memory can be more easily
  // aligned that way. should check cudaDevAttrHostRegisterSupported
  void *ptr;
  cudaError_t rc = cudaHostAlloc(&ptr, size, cudaHostAllocMapped);
  if (rc != cudaSuccess)
    {
      return nullptr;
    }

  rc = cudaMemset(ptr, 0, size);
  if (rc != cudaSuccess)
    {
      return nullptr;
    }

  return ptr;
}

int deallocate_shared(void *ptr)
{
  cudaError_t rc = cudaFreeHost(ptr);
  return rc == cudaSuccess ? 0 : 1;
}

void *device_ptr_from_host_ptr(void *host)
{
  void *device;
  unsigned int flags = 0;
  cudaError_t rc = cudaHostGetDevicePointer(&device, host, flags);
  if (rc != cudaSuccess)
    {
      return nullptr;
    }
  return device;
}

}  // namespace cuda_impl
}  // namespace allocator
}  // namespace hostrpc
