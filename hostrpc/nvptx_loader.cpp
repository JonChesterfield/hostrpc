#include "raiifile.hpp"
#include <cuda.h>
#include <cuda_runtime.h>

#include <memory>
#include <stdio.h>

#include "x64_host_ptx_client.hpp"

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

// Implementation api. This construct is a singleton.
namespace hostcall_ops
{
#if defined(__x86_64__)
void operate(hostrpc::page_t *page);
void clear(hostrpc::page_t *page);
#endif
#if defined(__AMDGCN__) || defined(__CUDACC__)
void pass_arguments(hostrpc::page_t *page, uint64_t data[8]);
void use_result(hostrpc::page_t *page, uint64_t data[8]);
#endif
}  // namespace hostcall_ops

namespace hostrpc
{
namespace x64_host_nvptx_client
{
struct fill
{
  static void call(hostrpc::page_t *page, void *dv)
  {
#if defined(__AMDGCN__)
    uint64_t *d = static_cast<uint64_t *>(dv);
    hostcall_ops::pass_arguments(page, d);
#else
    (void)page;
    (void)dv;
#endif
  };
};

struct use
{
  static void call(hostrpc::page_t *page, void *dv)
  {
#if defined(__AMDGCN__)
    uint64_t *d = static_cast<uint64_t *>(dv);
    hostcall_ops::use_result(page, d);
#else
    (void)page;
    (void)dv;
#endif
  };
};

struct operate
{
  static void call(hostrpc::page_t *page, void *)
  {
#if defined(__x86_64__)
    hostcall_ops::operate(page);
#else
    (void)page;
#endif
  }
};

struct clear
{
  static void call(hostrpc::page_t *page, void *)
  {
#if defined(__x86_64__)
    hostcall_ops::clear(page);
#else
    (void)page;
#endif
  }
};
}  // namespace x64_host_nvptx_client


using x64_nvptx_pair = hostrpc::x64_ptx_pair_T<
  hostrpc::size_runtime,
  x64_host_nvptx_client::fill, x64_host_nvptx_client::use,
    x64_host_nvptx_client::operate, x64_host_nvptx_client::clear,
    counters::client_nop, counters::server_nop>;

}  // namespace hostrpc

class hostcall_impl;
class hostcall
{
 public:
  hostcall();
  ~hostcall();
  bool valid() { return state_.get() != nullptr; }

  int enable_executable();
  int enable_queue();
  int spawn_worker();

  hostcall(const hostcall &) = delete;
  hostcall(hostcall &&) = delete;

 private:
  std::unique_ptr<hostcall_impl> state_;
};

int init(void *image)
{
  error_tracker t;

  t("cuInit", []() { return cuInit(0); });

  if (0) t("setDeviceFlags", []() {
                        cudaError_t err =cudaSetDeviceFlags(cudaDeviceMapHost);
                        fprintf(stderr, "setDeviceFlags returned %s\n", cudaGetErrorString(err));
                        if (err == cudaSuccess) { return CUDA_SUCCESS; }
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
    [&]() { return cuModuleGetFunction(&Func, Module, "_Z10cuda_hellov"); });


  hostrpc:: x64_nvptx_pair state (128); // hostrpc::size_runtime(128));
  
  t("cuLaunchKernel", [&]() {
    return cuLaunchKernel(/* kernel */ Func,
                          /*blocks per grid */ 1,
                          /*gridDimY */ 1,
                          /*gridDimZ*/ 1,
                          /* threads per block */ 32,
                          /* blockDimY */ 1,
                          /* blockDimZ */ 1,
                          /* sharedMemBytes */ 0, stream, nullptr, nullptr);
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
