#include "platform/detect.hpp"
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

#include "hsa.hpp"
#include "incbin.h"
#include <stdio.h>

INCBIN(run_on_hsa_example_so, "lib/run_on_hsa_example.gcn.so");

int load(hsa_executable_t* res)
{
  auto meta = hsa::parse_metadata(run_on_hsa_example_so_data,
                                  run_on_hsa_example_so_size);
  (void)meta;

  hsa_code_object_reader_t reader;
  hsa_status_t rc = hsa_code_object_reader_create_from_memory(
      run_on_hsa_example_so_data, run_on_hsa_example_so_size, &reader);
  if (rc != HSA_STATUS_SUCCESS)
    {
      bang();
      return 1;
    }

  hsa_executable_t ex;
  {
    hsa_profile_t profile =
        HSA_PROFILE_BASE;  // HIP uses full, vega claims 'base', unsure
    hsa_default_float_rounding_mode_t default_rounding_mode =
        HSA_DEFAULT_FLOAT_ROUNDING_MODE_DEFAULT;
    const char* options = 0;

    hsa_status_t rc =
        hsa_executable_create_alt(profile, default_rounding_mode, options, &ex);
    if (rc != HSA_STATUS_SUCCESS)
      {
        bang();
        return 1;
      }
  }

  rc = hsa::iterate_agents([&](hsa_agent_t agent) -> hsa_status_t {
    auto features = hsa::agent_get_info_feature(agent);
    if (features & HSA_AGENT_FEATURE_KERNEL_DISPATCH)
      {
        hsa_loaded_code_object_t code;
        rc = hsa_executable_load_agent_code_object(ex, agent, reader, NULL,
                                                   &code);
        if (rc != HSA_STATUS_SUCCESS)
          {
            bang();
            return rc;
          }
      }
    return HSA_STATUS_SUCCESS;
  });

  if (rc != HSA_STATUS_SUCCESS)
    {
      bang();
      return 1;
    }

  rc = hsa_executable_freeze(ex, NULL);
  if (rc != HSA_STATUS_SUCCESS)
    {
      bang();
      return 1;
    }

  {
    uint32_t vres;
    hsa_status_t rc = hsa_executable_validate(ex, &vres);
    if (rc != HSA_STATUS_SUCCESS)
      {
        bang();
        return 1;
      }

    if (vres != 0)
      {
        return 1;
      }
  }

  *res = ex;
  return 0;
}

int main()
{
  hsa::init live;
  example_type inst;
  inst.x = 11;
  inst.y = 3;
  inst.z = 0;

  hsa_executable_t ex;
  if (load(&ex) != 0)
    {
      return 1;
    }

  example_call(ex, &inst);

  fprintf(stderr, "Res %u ?= 33\n", inst.z);
}

#else

void example_call(example_type* ex) { ex->z = ex->x * ex->y; }

#endif
#endif
