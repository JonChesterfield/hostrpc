#include "x64_nvptx_pair.hpp"

#include "raiifile.hpp"
#include <cuda.h>
#include <cuda_runtime.h>

#include <memory>
#include <stdio.h>

#include "hostcall.hpp"
#include "x64_nvptx_pair.hpp"

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

struct error_tracker
{
  CUresult Err = CUDA_SUCCESS;
  template <typename F>
  void operator()(const char *ErrMsg, F f)
  {
    if (Err != CUDA_SUCCESS)
      {
        return;
      }

    Err = f();
    if (Err != CUDA_SUCCESS)
      {
        DP("%s\n", ErrMsg);
        CUDA_ERR_STRING(Err);
      }
  }

  explicit operator bool() const { return Err == CUDA_SUCCESS; }
};

#if defined(__x86_64__)
hostrpc::x64_nvptx_pair x64_nvptx_state(128);
#else
hostrpc::x64_nvptx_pair x64_nvptx_state;

#endif

int init(void *image)
{
  error_tracker t;

  t("cuInit", []() { return cuInit(0); });

  if (0)
    t("setDeviceFlags", []() {
      cudaError_t err = cudaSetDeviceFlags(cudaDeviceMapHost);
      fprintf(stderr, "setDeviceFlags returned %s\n", cudaGetErrorString(err));
      if (err == cudaSuccess)
        {
          return CUDA_SUCCESS;
        }
      return CUDA_ERROR_ILLEGAL_STATE;
    });

  int NumberOfDevices = 0;
  t("cuDeviceGetCount", [&]() {
    CUresult Err = cuDeviceGetCount(&NumberOfDevices);
    if (Err == CUDA_SUCCESS)
      {
        if (NumberOfDevices == 0)
          {
            return CUDA_ERROR_NO_DEVICE;
          }
      }
    return Err;
  });

  CUdevice Device;
  int DeviceId = 0;

  t("cuDeviceGet", [&]() { return cuDeviceGet(&Device, DeviceId); });

  // rtl does some get state calls here, maybe cuda doesn't clean up on exit?
  // TODO: Google CU_CTX_SCHED_BLOCKING_SYNC
  // todo: cuModuleUnload(module),  cuCtxDestroy(Context),

  CUcontext Context = nullptr;

  t("cuDevicePrimaryCtxRetain",
    [&]() { return cuDevicePrimaryCtxRetain(&Context, Device); });

  t("cuCtxSetCurrent", [&]() { return cuCtxSetCurrent(Context); });

  CUstream stream;

  t("cuStreamCreate",
    [&]() { return cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING); });

  // Some checking of number of threads

  t("warpsize", [&]() {
    int WarpSize;
    CUresult Err =
        cuDeviceGetAttribute(&WarpSize, CU_DEVICE_ATTRIBUTE_WARP_SIZE, Device);
    if (Err != CUDA_SUCCESS)
      {
        return Err;
      }
    if (WarpSize != 32)
      {
        return CUDA_ERROR_ILLEGAL_STATE;
      }

    return CUDA_SUCCESS;
  });

  CUmodule Module;
  t("cuModuleLoadDataEx",
    [&]() { return cuModuleLoadDataEx(&Module, image, 0, nullptr, nullptr); });

  CUfunction Func;
  t("cuModuleGetFunction",
    [&]() { return cuModuleGetFunction(&Func, Module, "__device_start"); });

  // hostrpc::size_runtime(128));

  t("cuLaunchKernel", [&]() {
    error_tracker t;

    int hostArgc = 0;
    void *hostArgv = NULL;  // todo: generalise the argv marshalling
    int hostRes[32];

    CUdeviceptr devArgc;
    CUdeviceptr devArgv;
    CUdeviceptr devRes;

    t("alloc", [&]() { return cuMemAlloc(&devArgc, sizeof(hostArgc)); });

    t("alloc", [&]() { return cuMemAlloc(&devArgv, sizeof(hostArgv)); });
    t("alloc", [&]() { return cuMemAlloc(&devRes, sizeof(hostRes)); });

    t("copy",
      [&]() { return cuMemcpyHtoD(devArgc, &hostArgc, sizeof(hostArgc)); });

    t("copy",
      [&]() { return cuMemcpyHtoD(devArgv, &hostArgv, sizeof(hostArgv)); });

    t("sync", [&]() { return cuStreamSynchronize(stream); });

    void *params[]{
        &devArgc,
        &devArgv,
        &devRes,
    };

    t("launch", [&]() {
      return cuLaunchKernel(/* kernel */ Func,
                            /*blocks per grid */ 1,
                            /*gridDimY */ 1,
                            /*gridDimZ*/ 1,
                            /* threads per block */ 32,
                            /* blockDimY */ 1,
                            /* blockDimZ */ 1,
                            /* sharedMemBytes */ 0, stream, params, nullptr);
    });

    t("copy",
      [&]() { return cuMemcpyDtoH(&hostRes, devRes, sizeof(hostRes)); });

    t("sync", [&]() { return cuStreamSynchronize(stream); });

    fprintf(stderr, "hostRes[0] = %u\n", hostRes[0]);

    t("free", [&]() { return cuMemFree(devArgc); });
    t("free", [&]() { return cuMemFree(devArgv); });
    t("free", [&]() { return cuMemFree(devRes); });

    return t.Err;
  });

  t("cuStreamSynchronize", [&]() { return cuStreamSynchronize(stream); });

  return t ? 0 : 1;
}

int main(int argc, char **argv)
{
  if (argc < 2)
    {
      fprintf(stderr, "Require at least one argument\n");
      return 1;
    }

  raiifile file(argv[1]);
  if (!file.mmapped_bytes)
    {
      fprintf(stderr, "Failed to open file %s\n", argv[1]);
      return 1;
    }

  int rc = init(file.mmapped_bytes);

  if (rc != 0)
    {
      return 1;
    }

  return 0;
}
