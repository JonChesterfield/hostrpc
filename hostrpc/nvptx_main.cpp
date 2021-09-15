#include "base_types.hpp"
#include "detail/platform_detect.hpp"
#include "x64_ptx_type.hpp"

// Implementation api. This construct is a singleton.
namespace hostcall_ops
{
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
hostrpc::x64_ptx_type<hostrpc::size_runtime<uint32_t>>::client_type
    *x64_nvptx_client_state = nullptr;

hostrpc::page_t scratch;
extern "C" __attribute__((visibility("default"))) int main(int argc,
                                                           char **argv)
{
  (void)argc;
  (void)argv;

  int s = 0;

  for (unsigned i = 0; i < 4; i++)
    {
      s += x64_nvptx_client_state
               ->rpc_invoke<hostrpc::fill_nop, hostrpc::use_nop>(
                   hostrpc::fill_nop{}, hostrpc::use_nop{});
    }

  return s;
}
#endif
