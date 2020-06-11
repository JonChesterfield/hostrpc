__attribute__((visibility("default"))) const char interp_section[]
    __attribute__((section(".interp"))) = "/home/amd/hostrpc/amdgcn_loader.exe";

int ocl_cast(int argc, __global void* argv);

static unsigned get_lane_id(void)
{
  return __builtin_amdgcn_mbcnt_hi(~0u, __builtin_amdgcn_mbcnt_lo(~0u, 0u));
}

kernel void device_entry(int argc, __global void* argv, __global int* res)
{
  // TODO: what does this do when res is non-uniform - picks one? May want
  // to return into res[64]
  res[get_lane_id()] = ocl_cast(argc, argv);
}
