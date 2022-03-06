#include "raiifile.hpp"
#include <cuda.h>
#include <cuda_runtime.h>

#include <thread>

#include <memory>
#include <stdio.h>

#include "x64_ptx_type.hpp"

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

  using SZ = hostrpc::size_runtime<uint32_t>;
  hostrpc::x64_ptx_type<SZ> x64_nvptx_state(SZ{128}, {}, {});

  {
    error_tracker t;

    CUdeviceptr devPtr;
    size_t devSz;
    CUresult err =
        cuModuleGetGlobal(&devPtr, &devSz, Module, "x64_nvptx_client_state");

    if (err == CUDA_SUCCESS)
      {
        fprintf(stderr, "found state at %llu\n", devPtr);

        CUdeviceptr mem;
        t("alloc", [&]() {
          return cuMemAlloc(&mem,
                            sizeof(hostrpc::x64_ptx_type<SZ>::client_type));
        });

        t("copy client", [&]() {
          return cuMemcpyHtoD(mem, &x64_nvptx_state,
                              sizeof(hostrpc::x64_ptx_type<SZ>::client_type));
        });

        t("copy pointer", [&]() { return cuMemcpyHtoD(devPtr, &mem, 8); });

        if (!t)
          {
            fprintf(stderr, "Failed to copy state to device\n");
          }
      }
    else
      {
        fprintf(stderr, "get symbol address failed, ret %u\n", err);
        exit(1);
      }
  }

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

    struct operate_test
    {
      void operator()(hostrpc::port_t, hostrpc::page_t *)
      {
        fprintf(stderr, "Invoked operate\n");
      }
    };
    struct clear_test
    {
      void operator()(hostrpc::port_t, hostrpc::page_t *)
      {
        fprintf(stderr, "Invoked clear\n");
      }
    };

    std::thread serv([&]() {
      uint32_t location = 0;

      for (unsigned i = 0; i < 16; i++)
        {
          bool r = x64_nvptx_state.server.rpc_handle<operate_test, clear_test>(
              operate_test{}, clear_test{}, &location);
          fprintf(stderr, "server ret %u\n", r);
          for (unsigned j = 0; j < 1000; j++)
            {
              platform::sleep();
            }
        }
    });

    t("launch", [&]() {
      fprintf(stderr, "Launching kernel\n");
      return cuLaunchKernel(/* kernel */ Func,
                            /*blocks per grid */ 1,
                            /*gridDimY */ 1,
                            /*gridDimZ*/ 1,
                            /* threads per block */ 32,
                            /* blockDimY */ 1,
                            /* blockDimZ */ 1,
                            /* sharedMemBytes */ 0, stream, params, nullptr);
    });

    fprintf(stderr, "Kernel launched\n");
    // times out here

    t("more sync", [&]() { return cuStreamSynchronize(stream); });
    fprintf(stderr, "Sync finished\n");

    t("copy",
      [&]() { return cuMemcpyDtoH(&hostRes, devRes, sizeof(hostRes)); });

    t("sync", [&]() { return cuStreamSynchronize(stream); });

    fprintf(stderr, "hostRes[0] = %u\n", hostRes[0]);

    serv.join();

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
