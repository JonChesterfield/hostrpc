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
  fprintf(stderr, "Called weak symbol: %s\n", __func__);
  return nullptr;
}
__attribute__((weak)) HOSTRPC_ANNOTATE int deallocate(void *)
{
  fprintf(stderr, "Called weak symbol: %s\n", __func__);
  return 1;
}

__attribute__((weak)) HOSTRPC_ANNOTATE int memsetzero_gpu(void *, size_t)
{
  fprintf(stderr, "Called weak symbol: %s\n", __func__);
  return 1;
}

}  // namespace hsa_impl

namespace cuda_impl
{
__attribute__((weak)) HOSTRPC_ANNOTATE void *allocate_shared(size_t)
{
  fprintf(stderr, "Called weak symbol: %s\n", __func__);
  return nullptr;
}
__attribute__((weak)) HOSTRPC_ANNOTATE int deallocate_shared(void *)
{
  fprintf(stderr, "Called weak symbol: %s\n", __func__);
  return 1;
}
__attribute__((weak)) HOSTRPC_ANNOTATE int memsetzero_gpu(void *, size_t)
{
  fprintf(stderr, "Called weak symbol: %s\n", __func__);
  return 1;
}
__attribute__((weak)) HOSTRPC_ANNOTATE void *device_ptr_from_host_ptr(void *)
{
  fprintf(stderr, "Called weak symbol: %s\n", __func__);
  return nullptr;
}
__attribute__((weak)) HOSTRPC_ANNOTATE void *host_ptr_from_device_ptr(void *)
{
  fprintf(stderr, "Called weak symbol: %s\n", __func__);
  return nullptr;
}
}  // namespace cuda_impl

}  // namespace allocator
}  // namespace hostrpc
