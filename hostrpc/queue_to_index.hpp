#ifndef QUEUE_TO_INDEX_HPP_INCLUDED
#define QUEUE_TO_INDEX_HPP_INCLUDED

#include <stddef.h>
#include <stdint.h>

#include "platform/detect.hpp"

// There is a finite number of HSA queues supported by the driver:
static const constexpr uint32_t MAX_NUM_DOORBELLS = 0x400;

// One can therefore map a pointer to a HSA queue to an integer in [0,
// MAX_NUM_DOORBELLS).
inline uint16_t queue_to_index(unsigned char *q);

#if HOSTRPC_HOST
#include "hsa.h"

// x64 is likely to have a hsa queue, and has no corresponding builtin
inline uint16_t queue_to_index(hsa_queue_t *queue)
{
  unsigned char *q = reinterpret_cast<unsigned char *>(queue);
  return queue_to_index(q);
}
#endif

#if defined(__AMDGCN__)
// gcn is likely to want to look up the hsa queue
inline uint16_t get_queue_index()
{
  __attribute__((address_space(4))) void *vq = __builtin_amdgcn_queue_ptr();
  unsigned char *q = (unsigned char *)vq;
  return queue_to_index(q);
}
#endif

inline uint16_t queue_to_index(unsigned char *q)
{
  // Given a pointer to the hsa queue,
  constexpr size_t doorbell_signal_offset = 16;
#if HOSTRPC_HOST
  // avoiding #include hsa.h on the gpu
  static_assert(
      offsetof(hsa_queue_t, doorbell_signal) == doorbell_signal_offset, "");
#endif

  uint64_t handle;
  __builtin_memcpy(&handle, q + doorbell_signal_offset, 8);
  char *sig = reinterpret_cast<char *>(handle);

  // The signal contains a kind at offset 0, expected to be -1 (non-legacy)
  int64_t kind;
  __builtin_memcpy(&kind, sig, 8);
#ifdef assert
  // it's an amd_signal_kind_t
  assert(kind == -1 || kind == -2);
#endif
  (void)kind;

  sig += 8;  // step over kind field

  // kind is probably a fixed function of architecture
  // todo: lift it from the gfxN macros
  if (kind == -1)
    {
      uint64_t ptr;
      __builtin_memcpy(&ptr, sig, 8);
      ptr >>= 3;
      ptr %= MAX_NUM_DOORBELLS;
      return static_cast<uint16_t>(ptr);
    }
  else
    {
      // This is not based on much, should test whether it works on gfx8 by
      // creating 0x400 queues and checking they all return a unique number
      uint32_t ptr;
      __builtin_memcpy(&ptr, sig, 4);
      ptr >>= 3;
      ptr %= MAX_NUM_DOORBELLS;
      return static_cast<uint16_t>(ptr);
    }
}

#if 0  // Does not compile for gfx8, hence using the intrinsic
static uint16_t get_queue_index_asm()
{
  static_assert(MAX_NUM_DOORBELLS < UINT16_MAX, "");
  uint32_t tmp0, tmp1;

  // Derived from mGetDoorbellId in amd_gpu_shaders.h, rocr
  // Using similar naming, exactly the same control flow.
  // This may be expensive enough to be worth caching or precomputing.
  uint32_t res;
  asm("s_mov_b32 %[tmp0], exec_lo\n\t"
      "s_mov_b32 %[tmp1], exec_hi\n\t"
      "s_mov_b32 exec_lo, 0x80000000\n\t"
      "s_sendmsg sendmsg(MSG_GET_DOORBELL)\n\t"
      "%=:\n\t"
      "s_nop 7\n\t"
      "s_bitcmp0_b32 exec_lo, 0x1F\n\t"
      "s_cbranch_scc0 %=b\n\t"
      "s_mov_b32 %[ret], exec_lo\n\t"
      "s_mov_b32 exec_lo, %[tmp0]\n\t"
      "s_mov_b32 exec_hi, %[tmp1]\n\t"
      : [ tmp0 ] "=&r"(tmp0), [ tmp1 ] "=&r"(tmp1), [ ret ] "=r"(res));

  res %= MAX_NUM_DOORBELLS;

  return static_cast<uint16_t>(res);
}
#endif

#endif
