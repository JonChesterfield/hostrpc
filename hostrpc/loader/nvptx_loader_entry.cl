
int __device_start_cast(int argc, __global void* argv);

#define WARPSIZE 32
static unsigned get_lane_id(void)
{
  return __nvvm_read_ptx_sreg_tid_x() & (WARPSIZE - 1);
}

kernel void __device_start(int argc, __global void* argv, __global int* res)
{
  res[get_lane_id()] = __device_start_cast(argc, argv);
}
