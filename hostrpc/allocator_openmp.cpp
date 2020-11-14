#include "allocator.hpp"

#include <cstdlib>
#include <omp.h>

namespace hostrpc
{
namespace allocator
{
namespace openmp_target_impl
{
HOSTRPC_ANNOTATE void *allocate(int device_num, size_t bytes)
{
  return omp_target_alloc(bytes, device_num);
}

HOSTRPC_ANNOTATE int deallocate(int device_num, void *ptr)
{
  omp_target_free(ptr, device_num);
  return 0;
}

}  // namespace openmp_target_impl

namespace openmp_shared_impl
{
HOSTRPC_ANNOTATE void *ctor()
{
  void *m = malloc(sizeof(omp_allocator_handle_t));
  if (m)
    {
      omp_allocator_handle_t *omp = new omp_allocator_handle_t;
      // spec says this takes a const array, but clang thinks it's mutable
      *omp = omp_init_allocator(omp_default_mem_space, 1,
                                (omp_alloctrait_t[1]){{
                                    .key = omp_atk_pinned,
                                    .value = omp_atv_true,
                                }});
      return reinterpret_cast<void *>(omp);
    }
  return m;
}

static omp_allocator_handle_t *get(void *state)
{
  return __builtin_launder(reinterpret_cast<omp_allocator_handle_t *>(state));
}

HOSTRPC_ANNOTATE void dtor(void *state)
{
  if (state)
    {
      omp_allocator_handle_t *alloc = get(state);
      omp_destroy_allocator(*alloc);
      free(alloc);
    }
}

HOSTRPC_ANNOTATE void *allocate(void *state, int, size_t bytes)
{
  if (state)
    {
      omp_allocator_handle_t *alloc = get(state);
      return omp_alloc(bytes, *alloc);
    }
  else
    {
      return nullptr;
    }
}

HOSTRPC_ANNOTATE int deallocate(void *state, int, void *ptr)
{
  if (state)
    {
      omp_allocator_handle_t *alloc = get(state);
      omp_free(ptr, *alloc);
    }
  return 0;
}

}  // namespace openmp_shared_impl
}  // namespace allocator
}  // namespace hostrpc
