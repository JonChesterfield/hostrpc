#include "common.hpp"
#include "platform.hpp"
#include "hostcall.hpp"

namespace hostcall_ops
{
#if defined(__x86_64__)
void operate(hostrpc::page_t *page)
{
  for (unsigned c = 0; c < 64; c++)
    {
      hostrpc::cacheline_t &line = page->cacheline[c];
      for (unsigned i = 0; i < 8; i++)
        {
          line.element[i] = 2 * (line.element[i] + 1);
        }
    }
}
#endif

#if defined __AMDGCN__
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


#if defined __AMDGCN__
extern "C" __attribute__((visibility("default"))) int main(int argc,
                                                           char **argv)
{
  (void)argc;
  (void)argv;
  int differ = 0;

  uint64_t initial_value = 7;

  hostrpc::cacheline_t line;
  hostrpc::cacheline_t expect;

  for (unsigned rep = 0; rep < 64000; rep++)
    {
      for (unsigned e = 0; e < 8; e++)
        {
          line.element[e] = initial_value + platform::get_lane_id() + e + rep;
          expect.element[e] = 2 * (line.element[e] + 1);
        }

      hostcall_client(&line.element[0]);

      for (unsigned e = 0; e < 8; e++)
        {
          differ += (line.element[e] != expect.element[e]);
        }
    }

  return differ;
}
#endif
