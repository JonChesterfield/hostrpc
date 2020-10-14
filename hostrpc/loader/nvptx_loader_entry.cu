__device__ extern "C" int main(int argc, char** argv);

#define WARPSIZE 32
__device__ static unsigned get_lane_id(void)
{
  return __nvvm_read_ptx_sreg_tid_x() & (WARPSIZE - 1);
}

__global__ extern "C" void __device_start(int argc, void* argv, int* res)
{
  res[get_lane_id()] = main(argc, (char**)argv);
}
