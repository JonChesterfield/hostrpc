#include "../../cxa_atexit.hpp"

extern "C" __attribute__((visibility("default"))) int main(int argc,
                                                           char** argv);

extern "C" int __device_start_cast(
    int argc, __attribute__((address_space(1))) void* vargv)
{
  // appears to be the usual way to request an addrspace cast
  char** argv = (char**)(vargv);
  return main(argc, argv);
}
