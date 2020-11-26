#include "allocator.hpp"

#include <cuda.h>
#include <cuda_runtime.h>

#include "detail/platform_detect.h"
#if !HOSTRPC_HOST
#error "allocator_cuda relies on the cuda host library"
#endif

#include <string.h>
#include <sys/mman.h>

namespace hostrpc
{
namespace allocator
{
namespace cuda_impl
{
HOSTRPC_ANNOTATE int memsetzero_gpu(void *memory, size_t bytes)
{
  cudaError_t rc;

  rc = cudaMemset(memory, 0, bytes);
  if (rc != cudaSuccess)
    {
      return 1;
    }

  rc = cudaDeviceSynchronize();
  if (rc != cudaSuccess)
    {
      return 1;
    }

  return 0;
}

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

#define VIA_MMAP 1

// cudaHostAlloc deadlocks when called from gpu kernel
// mmap alternative doesn't, but is asserting, and cudaMemset fails on it
// todo: recheck what the various cuda calls should be
void *allocate_shared(size_t size)
{
  size = (size + 4095) & ~((size_t)4095);
  // cudaHostRegister may be a better choice as the memory can be more easily
  // aligned that way. should check cudaDevAttrHostRegisterSupported
  fprintf(stderr, "call host alloc for %zu bytes\n", size);
  cudaError_t rc;

#if (VIA_MMAP)
  void *mapped = mmap(NULL, size, PROT_READ | PROT_WRITE,
                      MAP_PRIVATE | MAP_ANONYMOUS | MAP_LOCKED, -1, 0);

  if (mapped == MAP_FAILED)
    {
      return nullptr;
    }

  fprintf(stderr, "call host register\n");

  {
    // only have one device
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);

    if (!deviceProp.canMapHostMemory)
      {
        fprintf(stderr, "Device %d cannot map host memory!\n", 0);
        exit(EXIT_FAILURE);
      }
  }

  rc = cudaHostRegister(mapped, size, cudaHostRegisterMapped);
  if (rc != cudaSuccess)
    {
      fprintf(stderr, "host register ret %u\n", rc);
      return nullptr;
    }

  fprintf(stderr, "call memset on %p, %zu bytes\n", mapped, size);
  memset(mapped, 0, size);

  fprintf(stderr, "allocate+registered %p\n", mapped);
  return mapped;

#else

  void *ptr;

  rc = cudaHostAlloc(&ptr, size, cudaHostAllocMapped);
  if (rc != cudaSuccess)
    {
      return nullptr;
    }

  fprintf(stderr, "call memset\n");
  rc = cudaMemset(ptr, 0, size);
  if (rc != cudaSuccess)
    {
      return nullptr;
    }

  fprintf(stderr, "alloc shared done\n");

  return ptr;
#endif
}

int deallocate_shared(void *ptr)
{
  (void)ptr;
#if (VIA_MMAP)
  // leak, haven't tracked the size
  // munmap(ptr, size);
  return 0;
#else
  cudaError_t rc = cudaFreeHost(ptr);
  return rc == cudaSuccess ? 0 : 1;
#endif
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

void *host_ptr_from_device_ptr(void *device)
{
  cudaPointerAttributes attr;
  cudaError_t rc = cudaPointerGetAttributes(&attr, device);
  if (rc != cudaSuccess)
    {
      return nullptr;
    }
  return attr.hostPointer;
}

}  // namespace cuda_impl
}  // namespace allocator
}  // namespace hostrpc
