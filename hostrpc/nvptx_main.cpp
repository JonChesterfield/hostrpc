#include "base_types.hpp"
#include "detail/platform_detect.h"
#include "x64_ptx_type.hpp"

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

// __attribute__((visibility("default")))
// By value errors, 'Module has a nontrivial global ctor, which NVPTX does not
// support.'
hostrpc::x64_ptx_type::client_type *x64_nvptx_client_state = nullptr;

hostrpc::page_t scratch;
extern "C" __attribute__((visibility("default"))) int main(int argc,
                                                           char **argv)
{
  (void)argc;
  (void)argv;

  // won't work, just looking for the compile
  void *vp = static_cast<void *>(&scratch);

  // times out - no server running
  bool r = x64_nvptx_client_state
               ->rpc_invoke<hostrpc::fill_nop, hostrpc::use_nop, true>(vp, vp);

  return 0;
}
#endif
