#include "detail/platform_detect.hpp"

#include <stdint.h>
// may want to use __attribute__((address_space(3))) in openmp to restrict
// function to only LDS but that is not presently the case, works out OK via
// inlining and addrspace inference

extern "C"
{
  void __kmpc_impl_target_init();

  void __kmpc_impl_named_sync(uint32_t num_threads);
}

// void pteam_mem_barrier(uint32_t num_threads,
// /*__attribute__((address_space(3)))*/ uint32_t * barrier_state);

#include "EvilUnit.h"

unsigned get_lane_id()
{
  return __builtin_amdgcn_mbcnt_hi(~0u, __builtin_amdgcn_mbcnt_lo(~0u, 0u));
}

[[clang::loader_uninitialized]] __attribute__((address_space(3)))
uint32_t barrier_state;

__attribute__((visibility("default"))) unsigned main_workgroup_size_x =
    96;  // 128+32;

EVILUNIT_MAIN_MODULE()
{
  if (__builtin_amdgcn_workitem_id_x())
    {
      __kmpc_impl_target_init();
    }

  __builtin_amdgcn_s_barrier();  // syncthreads

  TEST("numbers")
  {
    if (0)
      printf("lane id %u / workitem %u / workgroup %u\n", get_lane_id(),
             __builtin_amdgcn_workitem_id_x(),
             __builtin_amdgcn_workgroup_id_x());
  }

  TEST("call barrier on all threads")
  {
    if (0) printf("before all\n");
    __kmpc_impl_named_sync(main_workgroup_size_x);
    __kmpc_impl_named_sync(main_workgroup_size_x);
    __kmpc_impl_named_sync(main_workgroup_size_x);
    if (0) printf("after all\n");
  }
}
