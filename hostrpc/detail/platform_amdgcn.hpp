#ifndef HOSTRPC_PLATFORM_AMDGCN_HPP_INCLUDED
#define HOSTRPC_PLATFORM_AMDGCN_HPP_INCLUDED

#ifndef HOSTRPC_PLATFORM_HPP_INCLUDED
#error "Expected to be included within platform.hpp"
#endif

#if !HOSTRPC_AMDGCN
#error "Expected HOSTRPC_AMDGCN"
#endif

namespace platform
{
HOSTRPC_ANNOTATE inline void sleep_briefly() { __builtin_amdgcn_s_sleep(0); }
HOSTRPC_ANNOTATE inline void sleep() { __builtin_amdgcn_s_sleep(100); }

HOSTRPC_ANNOTATE __attribute__((always_inline)) inline uint32_t get_lane_id()
{
  return __builtin_amdgcn_mbcnt_hi(~0u, __builtin_amdgcn_mbcnt_lo(~0u, 0u));
}

HOSTRPC_ANNOTATE __attribute__((always_inline)) inline bool is_master_lane()
{
  // TODO: 32 wide wavefront, consider not using raw intrinsics here
  uint64_t activemask = __builtin_amdgcn_read_exec();

  // TODO: check codegen for trunc lowest_active vs expanding lane_id
  // TODO: ffs is lifted from openmp runtime, looks like it should be ctz
  uint32_t lowest_active = __builtin_ffsl(activemask) - 1;
  uint32_t lane_id = get_lane_id();

  // TODO: readfirstlane(lane_id) == lowest_active?
  return lane_id == lowest_active;
}

HOSTRPC_ANNOTATE __attribute__((always_inline)) inline uint32_t
broadcast_master(uint32_t x)
{
  return __builtin_amdgcn_readfirstlane(x);
}

HOSTRPC_ANNOTATE inline uint32_t client_start_slot()
{
  // Ideally would return something < size
  // Attempt to distibute clients roughly across the array
  // compute unit currently executing the wave is a version of that

  // hip's runtime has macros that collide with these. That should be fixed in
  // hip, as 'HW_' is not a reserved namespace. Until then, bodge it here.
  enum
  {
    HRPC_HW_ID = 4,  // specify that the hardware register to read is HW_ID

    HRPC_HW_ID_CU_ID_SIZE = 4,    // size of CU_ID field in bits
    HRPC_HW_ID_CU_ID_OFFSET = 8,  // offset of CU_ID from start of register

    HRPC_HW_ID_SE_ID_SIZE = 2,     // sizeof SE_ID field in bits
    HRPC_HW_ID_SE_ID_OFFSET = 13,  // offset of SE_ID from start of register
  };
#define ENCODE_HWREG(WIDTH, OFF, REG) (REG | (OFF << 6) | ((WIDTH - 1) << 11))
  uint32_t cu_id = __builtin_amdgcn_s_getreg(
      ENCODE_HWREG(HRPC_HW_ID_CU_ID_SIZE, HRPC_HW_ID_CU_ID_OFFSET, HRPC_HW_ID));
  uint32_t se_id = __builtin_amdgcn_s_getreg(
      ENCODE_HWREG(HRPC_HW_ID_SE_ID_SIZE, HRPC_HW_ID_SE_ID_OFFSET, HRPC_HW_ID));
  return (se_id << HRPC_HW_ID_CU_ID_SIZE) + cu_id;
#undef ENCODE_HWREG
}

HOSTRPC_ANNOTATE __attribute__((always_inline)) inline void fence_acquire()
{
  __c11_atomic_thread_fence(__ATOMIC_ACQUIRE);
}
HOSTRPC_ANNOTATE __attribute__((always_inline)) inline void fence_release()
{
  __c11_atomic_thread_fence(__ATOMIC_RELEASE);
}

}  // namespace platform

#endif
