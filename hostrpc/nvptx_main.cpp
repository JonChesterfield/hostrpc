#if defined __NVPTX__

#include "x64_host_ptx_client.hpp"

extern "C" __attribute__((visibility("default"))) int main(int argc,
                                                           char **argv)
{
  (void)argc;
  (void)argv;

  return 42;
}

#endif
