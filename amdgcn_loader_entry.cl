// clang errors if this is called main

const char interp_section[] __attribute__((section(".interp"))) = "/home/amd/hostrpc/amdgcn_loader.exe";


__attribute__((noinline)) int vmain(int argc, char* __constant* argv)
{
  (void)argc;
  (void)argv;
  return 0;
}

// Not sure the __constant qualifier is worthwhile
kernel void device_entry(int argc, __constant void* vargv)
{
  char* __constant* argv = (char* __constant*)vargv;
  vmain(argc, argv);
}
