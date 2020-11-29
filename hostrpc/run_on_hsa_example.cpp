#include "detail/platform_detect.h"
#include "run_on_hsa.hpp"

// needs to parse as C for opencl, could use opencl++ instead
typedef struct 
{
  int x;
  int y;
  int z;
} example_type;


HOSTRPC_ENTRY_POINT(example_call, example_type)

#if !defined(__OPENCL_C_VERSION__)
#if HOSTRPC_HOST

#include <stdio.h>

int main()
{
  example_type inst;
  inst.x = 11;
  inst.y = 3;

  example_call(&inst);

  fprintf(stderr, "Res %u ?= 33\n", inst.z);

}

#else

void example_call(example_type * ex)
{
  ex->z = ex->x * ex->y;
}

#endif
#endif
