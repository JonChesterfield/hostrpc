#include "detail/platform_detect.hpp"

#include <stdint.h>
// may want to use __attribute__((address_space(3))) in openmp to restrict function to only LDS
void pteam_mem_barrier(uint32_t num_threads, /*__attribute__((address_space(3)))*/ uint32_t * barrier_state);

#include "EvilUnit.h"

typedef uint64_t __kmpc_impl_lanemask_t;

unsigned GetLaneId() {
  return __builtin_amdgcn_mbcnt_hi(~0u, __builtin_amdgcn_mbcnt_lo(~0u, 0u));
}

__kmpc_impl_lanemask_t __kmpc_impl_activemask() {
  return __builtin_amdgcn_read_exec();
}
uint32_t __kmpc_impl_ffs(uint32_t x) { return __builtin_ffs(x); }
uint32_t __kmpc_impl_ffs(uint64_t x) { return __builtin_ffsl(x); }

#define WARPSIZE 64

void pteam_mem_barrier(uint32_t num_threads,  uint32_t * barrier_state)
{
  __atomic_thread_fence(__ATOMIC_ACQUIRE);
  
  uint32_t num_waves = (num_threads + WARPSIZE - 1) / WARPSIZE ; // expected bug here

  printf("[%u/%u] in mem_barrier, thrds/waves %u %u\n",
         __builtin_amdgcn_workitem_id_x(),
         __builtin_amdgcn_workgroup_id_x(),
         num_threads, num_waves);

  
  // Partial barrier implementation for amdgcn.
  // Uses two 16 bit unsigned counters. One for the number of waves to have
  // reached the barrier, and one to count how many times the barrier has been
  // passed. These are packed in a single atomically accessed 32 bit integer.
  // Low bits for the number of waves, assumed zero before this call.
  // High bits to count the number of times the barrier has been passed.

  // precondition: num_waves != 0;
  // invariant: num_waves * WARPSIZE == num_threads;
  // precondition: num_waves < 0xffffu;

  // Increment the low 16 bits once, using the lowest active thread.
  uint64_t lowestActiveThread = __kmpc_impl_ffs(__kmpc_impl_activemask()) - 1;
  bool isLowest = true || GetLaneId() == lowestActiveThread;

  printf("[%u/%u] lAT %lu, isLowest %u, barrier_state %u\n",
         __builtin_amdgcn_workitem_id_x(),
         __builtin_amdgcn_workgroup_id_x(),
         lowestActiveThread,
         (unsigned)isLowest, __atomic_load_n(barrier_state, __ATOMIC_RELAXED));
  
  if (isLowest) {
    asm("// Before add");
    uint32_t load = __atomic_fetch_add(barrier_state, 1,
                                       __ATOMIC_ACQ_REL); // commutative

    asm("// After add");

    // Record the number of times the barrier has been passed
    uint32_t generation = load & 0xffff0000u;

    // both workgroups loaded zero. atomic_load gave 0, then
    // both workgroups fetch_add, and both returned 1 for the previous
    // value. so that doesn't seem right.
    
    printf("[%u/%u] loaded %u, generation 0x%x, lowbits 0x%x, reload %u\n",
         __builtin_amdgcn_workitem_id_x(),
         __builtin_amdgcn_workgroup_id_x(),
           load, generation, load & 0x0000ffffu, __atomic_load_n(barrier_state, __ATOMIC_RELAXED));

    
    if ((load & 0x0000ffffu) == (num_waves - 1)) {
      // Reached num_waves in low bits so this is the last wave.
      // Set low bits to zero and increment high bits
      
      load += 0x00010000u; // wrap is safe
      load &= 0xffff0000u; // because bits zeroed second

      // Reset the wave counter and release the waiting waves
      __atomic_store_n(barrier_state, load, __ATOMIC_RELAXED);
    } else {
      // more waves still to go, spin until generation counter changes
      do {
        __builtin_amdgcn_s_sleep(0);
        load = __atomic_load_n(barrier_state, __ATOMIC_RELAXED);
      } while ((load & 0xffff0000u) == generation);
    }
  }
  __atomic_thread_fence(__ATOMIC_RELEASE);
}



static inline uint32_t get_lane_id()
{
  return __builtin_amdgcn_mbcnt_hi(~0u, __builtin_amdgcn_mbcnt_lo(~0u, 0u));
}


[[clang::loader_uninitialized]]
__attribute__((address_space(3)))
uint32_t barrier_state;


__attribute__((visibility("default"))) unsigned main_workgroup_size_x = 128;
__attribute__((visibility("default"))) unsigned main_grid_size_x = 128;



extern "C"
__attribute__((used))
void test_pteam_mem_barrier(uint32_t num_threads)
{
  uint32_t * tmp = (uint32_t*) &barrier_state;
  pteam_mem_barrier(num_threads, tmp);
}


EVILUNIT_MAIN_MODULE()
{
  __atomic_store_n(&barrier_state, 0u, __ATOMIC_RELEASE);

  __builtin_amdgcn_s_barrier(); // syncthreads
  
  TEST("numbers")
    {
      printf("lane id %u / workitem %u / workgroup %u\n", get_lane_id(), __builtin_amdgcn_workitem_id_x(), __builtin_amdgcn_workgroup_id_x());
    }

  TEST("call barrier on all threads")
    {
      printf("before all\n");
      test_pteam_mem_barrier(main_grid_size_x);
      printf("after all\n");
    }

  TEST("call barrier on wg 0 threads")
    {
      if (__builtin_amdgcn_workgroup_id_x() == 0)
        {
      printf("before wg0\n");
          test_pteam_mem_barrier(main_workgroup_size_x);
      printf("after wg0\n");
        }
    }

}
