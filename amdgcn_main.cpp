
#include "x64_host_amdgcn_client_api.hpp"

// Example.
extern "C" __attribute__((visibility("default"))) int main(int argc,
                                                           char **argv)
{
  (void)argc;
  (void)argv;

  uint64_t data[8] = {0, 1, 2, 3, 4, 5, 6, 7};
  uint64_t want[8];
  for (unsigned i = 0; i < 8; i++)
    {
      want[i] = data[i] * data[i];
    }
  hostcall_client_async(data);
  int differ = 8;
  for (unsigned i = 0; i < 8; i++)
    {
      if (want[i] == data[i]) differ--;
    }

  return differ;
}
