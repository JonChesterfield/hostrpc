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

  uint64_t initial_value = 7;

  hostrpc::cacheline_t line;
  hostrpc::cacheline_t expect;

  for (unsigned e = 0; e < 8; e++)
    {
      line.element[e] = initial_value + platform::get_lane_id() + e;
      expect.element[e] = 2 * (line.element[e] + 1);
    }

  hostcall_client(&line.element[0]);

  for (unsigned e = 0; e < 8; e++)
    {
      differ += (line.element[e] != expect.element[e]);
    }

  // Calling a second time shows curious properties
  // O0: Memory violation raised by hsa. Either:
  //     HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION:  The agent attempted to
  //     access memory beyond the largest legal address
  // or
  // Memory access fault by GPU node-4 (Agent handle: 0x7331d0) on address
  // 0x7f67fda1b000. Reason: Unknown.
  // O1: Success
  // O2: Runs but gets the answer wrong

  if (1)
    {
      for (unsigned e = 0; e < 8; e++)
        {
          line.element[e] = initial_value + platform::get_lane_id() + e;
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
