extern "C" int main(int argc, char** argv);

extern "C" int ocl_cast(int argc, __attribute__((address_space(1))) void* vargv)
{
  // appears to be the usual way to request an addrspace cast
  char** argv = (char**)(vargv);
  return main(argc, argv);
}
