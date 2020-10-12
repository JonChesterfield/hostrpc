#include <cuda.h>
#include <sys/mman.h>  // mmap and fstat
#include <sys/stat.h>

#include <stdio.h>

#define DEBUGP(prefix, ...)             \
  {                                     \
    fprintf(stderr, "%s --> ", prefix); \
    fprintf(stderr, __VA_ARGS__);       \
  }

static int DebugLevel = 1;

#define GETNAME2(name) #name
#define GETNAME(name) GETNAME2(name)
#define DP(...)                                                       \
  do                                                                  \
    {                                                                 \
      if (DebugLevel > 0)                                             \
        {                                                             \
          DEBUGP("Target " GETNAME(TARGET_NAME) " RTL", __VA_ARGS__); \
        }                                                             \
    }                                                                 \
  while (false)

// Utility for retrieving and printing CUDA error string.
#define CUDA_ERR_STRING(err)                                                   \
  do                                                                           \
    {                                                                          \
      if (DebugLevel > 0)                                                      \
        {                                                                      \
          const char *errStr;                                                  \
          cuGetErrorString(err, &errStr);                                      \
          DEBUGP("Target " GETNAME(TARGET_NAME) " RTL", "CUDA error is: %s\n", \
                 errStr);                                                      \
        }                                                                      \
    }                                                                          \
  while (false)

namespace
{
bool checkResult(CUresult Err, const char *ErrMsg)
{
  if (Err == CUDA_SUCCESS) return true;

  DP("%s", ErrMsg);
  CUDA_ERR_STRING(Err);
  return false;
}
}  // namespace

void init(void *image)
{
  int NumberOfDevices;

  CUresult Err = cuInit(0);

  if (!checkResult(Err, "Error returned from cuInit\n"))
    {
      return;
    }

  Err = cuDeviceGetCount(&NumberOfDevices);
  if (!checkResult(Err, "Error returned from cuDeviceGetCount\n")) return;

  if (NumberOfDevices == 0)
    {
      DP("There are no devices supporting CUDA.\n");
      return;
    }

  fprintf(stderr, "Found %d devices\n", NumberOfDevices);

  CUdevice Device;
  int DeviceId = 0;
  Err = cuDeviceGet(&Device, DeviceId);
  if (!checkResult(Err, "Error returned from cuDeviceGet\n")) return;

  // rtl does some get state calls here, maybe cuda doesn't clean up on exit?
  // TODO: Google CU_CTX_SCHED_BLOCKING_SYNC
  CUcontext Context = nullptr;

  Err = cuDevicePrimaryCtxRetain(&Context, Device);
  if (!checkResult(Err, "Error returned from cuDevicePrimaryCtxRetain\n"))
    return;

  Err = cuCtxSetCurrent(Context);
  if (!checkResult(Err, "Error returned from cuCtxSetCurrent\n")) return;

  CUstream stream;
  Err = cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING);
  if (!checkResult(Err, "Error returned from cuStreamCreate")) return;

  // Some checking of number of threads

  int WarpSize;
  Err = cuDeviceGetAttribute(&WarpSize, CU_DEVICE_ATTRIBUTE_WARP_SIZE, Device);
  if (Err != CUDA_SUCCESS || WarpSize != 32)
    {
      return;
    }

  CUmodule Module;
  Err = cuModuleLoadDataEx(&Module, image, 0, nullptr, nullptr);
  if (!checkResult(Err, "Error returned from cuModuleLoadDataEx\n")) return;

  CUfunction Func;
  Err = cuModuleGetFunction(&Func, Module, "_Z10cuda_hellov");
  if (!checkResult(Err, "Error returned from cuModuleGetFunction\n")) return;

  Err = cuLaunchKernel(/* kernel */ Func,
                       /*blocks per grid */ 1,
                       /*gridDimY */ 1,
                       /*gridDimZ*/ 1,
                       /* threads per block */ 32,
                       /* blockDimY */ 1,
                       /* blockDimZ */ 1,
                       /* sharedMemBytes */ 0, stream, nullptr, nullptr);
  if (!checkResult(Err, "Error returned from cuLaunchKernel\n")) return;

  Err = cuStreamSynchronize(stream);
  if (!checkResult(Err, "Error returned from stream synchronize\n")) return;

  fprintf(stderr, "Reached end of init\n");
}

int main(int argc, char **argv)
{
  if (argc < 2)
    {
      fprintf(stderr, "Require at least one argument\n");
      return 1;
    }

  // Load argv[1]
  FILE *fh = fopen(argv[1], "rb");  // todo: close
  int fn = fh ? fileno(fh) : -1;
  if (fn < 0)
    {
      fprintf(stderr, "Failed to open file %s\n", argv[1]);
      return 1;
    }

  void *mmapped_bytes = nullptr;
  size_t mmapped_length = 0;
  {
    struct stat buf;
    int rc = fstat(fn, &buf);
    if (rc == 0)
      {
        size_t l = buf.st_size;
        void *m = mmap(NULL, l, PROT_READ, MAP_PRIVATE, fn, 0);
        if (m != MAP_FAILED)
          {
            mmapped_bytes = m;
            mmapped_length = l;
          }
      }
  }

  if (!mmapped_bytes)
    {
      return 1;
    }

  init(mmapped_bytes);

  if (mmapped_bytes != nullptr)
    {
      munmap(mmapped_bytes, mmapped_length);
    }
}
