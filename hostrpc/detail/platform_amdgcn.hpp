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
namespace
{
HOSTRPC_ANNOTATE constexpr uint64_t desc::native_width()
{
#ifndef __AMDGCN_WAVEFRONT_SIZE
#error "Expected __AMDGCN_WAVEFRONT_SIZE definition"
#endif
  return __AMDGCN_WAVEFRONT_SIZE;
}
static_assert(desc::native_width() == 32 || desc::native_width() == 64, "");

HOSTRPC_ANNOTATE uint64_t desc::active_threads()
{
  return (desc::native_width() == 64) ? __builtin_amdgcn_read_exec()
                                      : __builtin_amdgcn_read_exec_lo();
}

inline HOSTRPC_ANNOTATE void sleep_briefly() { __builtin_amdgcn_s_sleep(0); }
inline HOSTRPC_ANNOTATE void sleep() { __builtin_amdgcn_s_sleep(100); }

HOSTRPC_ANNOTATE __attribute__((always_inline)) inline uint32_t get_lane_id()
{
  return __builtin_amdgcn_mbcnt_hi(~0u, __builtin_amdgcn_mbcnt_lo(~0u, 0u));
}

template <typename T>
inline HOSTRPC_ANNOTATE uint32_t get_master_lane_id(T active_threads)
{
  auto f = active_threads.findFirstSet();
  return f.template subtract<1>();
}

template <typename T>
HOSTRPC_ANNOTATE __attribute__((always_inline)) inline uint32_t
broadcast_master(T, uint32_t x)
{
  // reads from lowest set bit in exec mask
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

}  // namespace
}  // namespace platform

#endif
