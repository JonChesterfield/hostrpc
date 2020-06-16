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
