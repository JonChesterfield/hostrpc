__attribute__((visibility("default"))) const char interp_section[]
    __attribute__((section(".interp"))) = "/home/amd/hostrpc/amdgcn_loader.exe";

int __device_start_cast(int argc, __global void* argv);

static unsigned get_lane_id(void)
{
  return __builtin_amdgcn_mbcnt_hi(~0u, __builtin_amdgcn_mbcnt_lo(~0u, 0u));
}

kernel void __device_start(int argc, __global void* argv, __global int* res)
{
  res[get_lane_id()] = __device_start_cast(argc, argv);
}
