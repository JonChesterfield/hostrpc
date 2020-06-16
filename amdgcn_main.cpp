#include "common.hpp"
#include "platform.hpp"
#include "x64_host_amdgcn_client_api.hpp"
// Example.

extern "C" __attribute__((visibility("default"))) int main(int argc,
                                                           char **argv)
{
  (void)argc;
  (void)argv;
  int differ = 0;

  int tmp0, tmp1;

  // Derived from mGetDoorbellId in amd_gpu_shaders.h, rocr
  // Using similar naming, exactly the same control flow.
  uint32_t SENDMSG_M0_DOORBELL_ID_BITS = 12;
  uint32_t SENDMSG_M0_DOORBELL_ID_MASK =
      ((1 << SENDMSG_M0_DOORBELL_ID_BITS) - 1);

  uint32_t MAX_NUM_DOORBELLS_MASK = ((1 << 10) - 1);

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

  res &= SENDMSG_M0_DOORBELL_ID_MASK;  // Doorbell index

  // Rocr masks with this, but values > 1024 are returned from the above
  if (1)
    {
      res &= MAX_NUM_DOORBELLS_MASK;  // Reduce further
    }

  return res;

  uint64_t initial_value = 7;

  hostrpc::cacheline_t line;
  hostrpc::cacheline_t expect;

  for (unsigned rep = 0; rep < 64000; rep++)
    {
      for (unsigned e = 0; e < 8; e++)
        {
          line.element[e] = initial_value + platform::get_lane_id() + e + rep;
          expect.element[e] = 2 * (line.element[e] + 1);
        }

      hostcall_client(&line.element[0]);

      for (unsigned e = 0; e < 8; e++)
        {
          differ += (line.element[e] != expect.element[e]);
        }
    }

  return differ;
}
