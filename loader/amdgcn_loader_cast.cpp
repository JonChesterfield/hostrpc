extern "C" __attribute__((visibility("default"))) int main(int argc,
                                                           char** argv);

extern "C" int __device_start_cast(int argc, __attribute__((address_space(1))) void* vargv)
{
  // appears to be the usual way to request an addrspace cast
  char** argv = (char**)(vargv);
  return main(argc, argv);
}

// Toolchain doesn't seem totally set up. Atexit gets called from global
// constructors, but global constructors don't actually get run on amdgcn/hsa.
// 'Implementing' here for now.
extern "C" int __cxa_atexit(void (*)(void*), void*, void*) { return 0; }
