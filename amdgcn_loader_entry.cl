const char interp_section[] __attribute__((section(".interp"))) =
    "/home/amd/hostrpc/amdgcn_loader.exe";

int ocl_cast(int argc, __global void* argv);

kernel void device_entry(int argc, __global void* argv, __global int* res)
{
  *res = ocl_cast(argc, argv);
}
