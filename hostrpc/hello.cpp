#include <cuda.h>

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

void init()
{
  CUresult Err = cuInit(0);
  int NumberOfDevices;

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
}

int main() { init(); }
