#include "base_types.hpp"
#include "detail/platform_detect.h"
#include "x64_host_ptx_client.hpp"

// Implementation api. This construct is a singleton.
namespace hostcall_ops
{
#if (HOSTRPC_HOST)
void operate(hostrpc::page_t *page)
{
  (void)page;
  fprintf(stderr, "Hit operate!\n");
}
void clear(hostrpc::page_t *page)
{
  (void)page;
  fprintf(stderr, "Hit clear!\n");
};
#endif
#if (HOSTRPC_GPU)
// from openmp_hostcall (amdgcn)
void pass_arguments(hostrpc::page_t *page, uint64_t d[8])
{
  hostrpc::cacheline_t *line = &page->cacheline[platform::get_lane_id()];
  for (unsigned i = 0; i < 8; i++)
    {
      line->element[i] = d[i];
    }
}
void use_result(hostrpc::page_t *page, uint64_t d[8])
{
  hostrpc::cacheline_t *line = &page->cacheline[platform::get_lane_id()];
  for (unsigned i = 0; i < 8; i++)
    {
      d[i] = line->element[i];
    }
}

#endif
}  // namespace hostcall_ops

#if (HOSTRPC_GPU)
extern "C" __attribute__((visibility("default"))) int main(int argc,
                                                           char **argv)
{
  (void)argc;
  (void)argv;

  return 42;
}
#endif
