#if defined __OPENCL__
// called by test
void example(void);
kernel void __device_example(void) { example(); }
#else

#include "cxa_atexit.hpp"
#include "detail/platform_detect.hpp"
#include "x64_gcn_type.hpp"

#undef printf
#include "hostrpc_printf.h"

#if (HOSTRPC_AMDGCN)
extern "C" void example(void)
{
  unsigned id = platform::get_lane_id();

  uint32_t port = piecewise_print_start(
      "some format %u too "
      "loffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
      "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
      "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
      "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
      "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
      "ffffffffffffffffffffffffffffng with %s fields\n");
  piecewise_pass_element_uint64(port, 42);
  piecewise_pass_element_cstr(port, "stringy");
  piecewise_print_end(port);

  printf("printf a u64 %lu\n", UINT64_C(111));

  print_string("string with %s formatting %% chars\n");

  // this is interesting because the first generates more packets
  // than the second.
  if (id % 3 == 0)
    {
      print_string(
          "this string is somewhat too long to fit in a single buffer so "
          "splits across several\n");
    }
  else
    {
      print_string("mostly short though\n");
    }

  platform::sleep();

  printf("test %lu call\n", 42, 0, 0);

  if (id % 2) printf("second %lu/%lu/%lu call\n", 101, 5, 2);
}

#endif

#if (HOSTRPC_HOST)
// tests

#undef printf
#include "hsa.hpp"
#include "incbin.h"
#include <stdio.h>
#include <vector>

INCBIN(x64_gcn_debug_so, "x64_gcn_debug.gcn.so");

int main()
{
  std::vector<hsa_agent_t> gpus = hsa::find_gpus();
  for (auto g : gpus)
    {
      hsa_queue_t *queue = hsa::create_queue(g);
      auto ex =
          hsa::executable(g, x64_gcn_debug_so_data, x64_gcn_debug_so_size);
      if (!ex.valid())
        {
          printf("gpu %lu ex not valid\n", g.handle);
          return 1;
        }

      int rc = hostrpc_print_enable_on_hsa_agent(ex, g);
      if (rc != 0)
        {
          printf("gpu %lu, enable -> %u\n", g.handle, rc);
          return 1;
        }

      hsa_signal_t sig;
      if (hsa_signal_create(1, 0, NULL, &sig) != HSA_STATUS_SUCCESS)
        {
          return 1;
        }

      if (hsa::launch_kernel(ex, queue, "__device_example.kd", 0, 0, sig) != 0)
        {
          return 1;
        }

      do
        {
        }
      while (hsa_signal_wait_acquire(sig, HSA_SIGNAL_CONDITION_EQ, 0,
                                     5000 /*000000*/,
                                     HSA_WAIT_STATE_ACTIVE) != 0);

      hsa_signal_destroy(sig);
      hsa_queue_destroy(queue);

      return 0;  // skip the second gpu
    }
}

#endif

#endif
