#include "detail/platform_detect.hpp"
#include "pool_interface.hpp"

POOL_INTERFACE_BOILERPLATE_AMDGPU(example, 32);

#if !defined(__OPENCL_C_VERSION__)
#include "detail/platform.hpp"

#if HOSTRPC_AMDGCN

#define ENCODE_HWREG(WIDTH, OFF, REG) (REG | (OFF << 6) | ((WIDTH - 1) << 11))
enum {
  HW_ID = 4, // specify that the hardware register to read is HW_ID


  HW_ID_WAVE_ID_SIZE = 4,
  HW_ID_WAVE_ID_OFFSET = 0,

  HW_ID_SIMD_ID_SIZE = 2,
  HW_ID_SIMD_ID_OFFSET = 4,
  
  HW_ID_CU_ID_SIZE = 4,   // size of CU_ID field in bits
  HW_ID_CU_ID_OFFSET = 8, // offset of CU_ID from start of register

  HW_ID_SE_ID_SIZE = 2,    // sizeof SE_ID field in bits
  HW_ID_SE_ID_OFFSET = 13, // offset of SE_ID from start of register
};

 uint32_t __kmpc_impl_smid() {
  uint32_t cu_id = __builtin_amdgcn_s_getreg(
      ENCODE_HWREG(HW_ID_CU_ID_SIZE, HW_ID_CU_ID_OFFSET, HW_ID));
  uint32_t se_id = __builtin_amdgcn_s_getreg(
      ENCODE_HWREG(HW_ID_SE_ID_SIZE, HW_ID_SE_ID_OFFSET, HW_ID));
  return (se_id << HW_ID_CU_ID_SIZE) + cu_id;
}

 uint32_t __kmpc_impl_wave() {
  return __builtin_amdgcn_s_getreg(
      ENCODE_HWREG(HW_ID_WAVE_ID_SIZE, HW_ID_WAVE_ID_OFFSET, HW_ID));
}

uint32_t __kmpc_impl_simd() {
  return __builtin_amdgcn_s_getreg(
      ENCODE_HWREG(HW_ID_SIMD_ID_SIZE, HW_ID_SIMD_ID_OFFSET, HW_ID));
}


uint32_t example::run(uint32_t state)
{
  if (1)
    if (platform::is_master_lane())
      printf("[%u.%u.%u] run %u from %u (of %u/%u)\n", __kmpc_impl_wave(),__kmpc_impl_simd(),__kmpc_impl_smid(), state, get_current_uuid(), alive(),
             requested());

  platform::sleep_briefly();
  return state + 1;
}
#endif

#if HOSTRPC_HOST
#include "hsa.hpp"
#include "incbin.h"
#include "launch.hpp"

INCBIN(pool_example_amdgpu_so, "pool_example_amdgpu.gcn.so");

// need to split enable print off from the macro
#define WITH_PRINTF 1

#if WITH_PRINTF
#include "hostrpc_printf_enable.h"
#endif

int main_with_hsa()
{
  hsa_agent_t kernel_agent = hsa::find_a_gpu_or_exit();
  fprintf(stderr, "Using agent %lu\n", kernel_agent.handle);
  auto ex = hsa::executable(kernel_agent, pool_example_amdgpu_so_data,
                            pool_example_amdgpu_so_size);
  if (!ex.valid())
    {
      fprintf(stderr, "Failed to load executable %s\n",
              "pool_example_amdgpu.gcn.so");
      exit(1);
    }
  else
    {
      fprintf(stderr, "Loaded executable %s\n", "pool_example_amdgpu.gcn.so");
    }

#if WITH_PRINTF
  if (hostrpc_print_enable_on_hsa_agent(ex, kernel_agent) != 0)
    {
      fprintf(stderr, "Failed to create host printf thread\n");
      exit(1);
    }
#endif
  
  hsa_queue_t *queue = hsa::create_queue(kernel_agent);
  if (!queue)
    {
      fprintf(stderr, "Failed to create queue\n");
      exit(1);
    }

  {
    unsigned char *base;
    __builtin_memcpy(&base, (char *)queue + 8, 8);
    uint32_t size;
    __builtin_memcpy(&size, (char *)queue + 24, 4);

    fprintf(stderr, "Queue is %p, packet base %p, size %u\n", queue, base,
            size);
  }

  fprintf(stderr, "Call initialize\n");
  example::initialize(ex, queue);

  fprintf(stderr, "Call bootstrap\n");
  example::bootstrap_entry(1);
    
  // leave them running for a while 
  usleep(50000000);

  fprintf(stderr, "Call teardown\n");

  example::teardown();

  example::finalize();

  {
    hsa_status_t rc = hsa_queue_destroy(queue);
    if (rc != HSA_STATUS_SUCCESS)
      {
        fprintf(stderr, "Failed to destroy queue: %s\n",
                hsa::status_string(rc));
      } else {
      fprintf(stderr, "Queue destroyed\n");
    }
  }

  fprintf(stderr, "Finished\n");
  return 0;
}

int main()
{
  fprintf(stderr, "In main\n");
  hsa::init hsa_state;
  return main_with_hsa();
}

#endif
#endif
