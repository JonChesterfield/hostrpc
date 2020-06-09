
#include "x64_host_amdgcn_client_api.hpp"

// Example.
extern "C" __attribute__((visibility("default"))) int main(int argc,
                                                           char **argv)
{
  (void)argc;
  (void)argv;
  uint64_t data[8] = {0, 1, 2, 3, 4, 5, 6, 7};

  hostcall_client_async(data);

  return 0;
}
