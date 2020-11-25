#include "allocator.hpp"
#include "openmp_plugins.hpp"

#include <cstdlib>
#include <omp.h>

namespace hostrpc
{
namespace allocator
{
// Chooses hsa or cuda at runtime, but a system may not have both present
namespace hsa_impl
{
__attribute__((weak)) HOSTRPC_ANNOTATE void *allocate_fine_grain(size_t)
{
  fprintf(stderr, "Called weak symbol\n");
  return nullptr;
}
__attribute__((weak)) HOSTRPC_ANNOTATE int deallocate(void *)
{
  fprintf(stderr, "Called weak symbol\n");
  return 1;
}

__attribute__((weak)) HOSTRPC_ANNOTATE int memsetzero_gpu(void *, size_t)
{
  fprintf(stderr, "Called weak symbol\n");
  return 1;
}

}  // namespace hsa_impl

namespace cuda_impl
{
__attribute__((weak)) HOSTRPC_ANNOTATE void *allocate_shared(size_t)
{
  fprintf(stderr, "Called weak symbol\n");

  return nullptr;
}
__attribute__((weak)) HOSTRPC_ANNOTATE int deallocate_shared(void *)
{
  fprintf(stderr, "Called weak symbol\n");
  return 1;
}
__attribute__((weak)) HOSTRPC_ANNOTATE int memsetzero_gpu(void *, size_t)
{
  fprintf(stderr, "Called weak symbol\n");
  return 1;
}
}  // namespace cuda_impl

namespace openmp_impl
{
HOSTRPC_ANNOTATE void *allocate_device(int device_num, size_t bytes)
{
  plugins p = hostrpc::find_plugins();
  if ((p.amdgcn + p.nvptx) != 1)
    {
      return 0;
    }

  bytes = 4 * ((bytes + 3) / 4);
  void *res = omp_target_alloc(bytes, device_num);

  // This probably works if the source is compiled as openmp. Compiling it
  // as c++ definitely doesn't work, and this doesn't work in host fallback
#if 0
  // zero it. should do this in parallel, simple for now.
  size_t words = bytes / sizeof(uint32_t);

#pragma omp target map(to              \
                       : words, tofrom \
                       : res [0:words] is_device_ptr(res) device(device_num)
  {
    uint32_t *r = (uint32_t *)res;
    for (size_t i = 0; i < words; i++)
      {
        r[i] = 0;
      }
  }

#endif

  if (res)
    {
      if (p.amdgcn)
        {
          if (hsa_impl::memsetzero_gpu(res, bytes) == 0)
            {
              return res;
            }
        }
      if (p.nvptx)
        {
          if (cuda_impl::memsetzero_gpu(res, bytes) == 0)
            {
              return res;
            }
        }

      deallocate_device(device_num, res);
    }

  return 0;
}

HOSTRPC_ANNOTATE int deallocate_device(int device_num, void *ptr)
{
  omp_target_free(ptr, device_num);
  return 0;
}

HOSTRPC_ANNOTATE void *allocate_shared(size_t bytes)
{
  plugins p = hostrpc::find_plugins();
  if ((p.amdgcn + p.nvptx) != 1)
    {
      printf("allocate shared: no plugins\n");
      return 0;
    }
  if (p.amdgcn)
    {
      return hsa_impl::allocate_fine_grain(bytes);
    }
  if (p.nvptx)
    {
      return cuda_impl::allocate_shared(bytes);
    }
  return 0;
}
HOSTRPC_ANNOTATE int deallocate_shared(void *ptr)
{
  plugins p = hostrpc::find_plugins();
  if (ptr)
    {
      if (p.amdgcn)
        {
          return hsa_impl::deallocate(ptr);
        }
      if (p.nvptx)
        {
          return cuda_impl::deallocate_shared(ptr);
        }
    }
  return 0;
}

}  // namespace openmp_impl

}  // namespace allocator
}  // namespace hostrpc
