#include <stdio.h>

#if defined __NVPTX__

extern "C" __attribute__((visibility("default"))) int main(int argc,
                                                           char **argv)
{
  (void)argc;
  (void)argv;

  return 42;
}

#endif
