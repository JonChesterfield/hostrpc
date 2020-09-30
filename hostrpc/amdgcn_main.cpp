#include "detail/platform.hpp"
#include "hostcall.hpp"

namespace hostcall_ops
{
#if defined(__x86_64__)
void operate(hostrpc::page_t *page)
{
  const bool verbose = false;
  if (verbose)
    {
      printf("Called operate\n");
    }
  for (unsigned c = 0; c < 64; c++)
    {
      hostrpc::cacheline_t &line = page->cacheline[c];
      auto l = [&](uint64_t idx) { return line.element[idx]; };

      if (verbose)
        {
          printf("line[%u]: %lu %lu %lu %lu %lu %lu %lu %lu\n", c, l(0), l(1),
                 l(2), l(3), l(4), l(5), l(6), l(7));
        }
      for (unsigned i = 0; i < 8; i++)
        {
          line.element[i] = 2 * (line.element[i] + 1);
        }
    }
}

void clear(hostrpc::page_t *page)
{
  for (unsigned c = 0; c < 64; c++)
    {
      hostrpc::cacheline_t &line = page->cacheline[c];
      for (unsigned i = 0; i < 8; i++)
        {
          line.element[i] = UINT64_MAX;
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

extern "C"
{
  __attribute__((used)) void init_page(hostrpc::page_t *page, uint64_t v)
  {
    hostrpc::cacheline_t *line = &page->cacheline[platform::get_lane_id()];
    for (unsigned i = 0; i < 8; i++)
      {
        line->element[i] = v;
      }
  }

#if 0
  init_page:                              ; @init_page
init_page$local:
; %bb.0:                                ; %entry
	s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)

        // Set v4 = lane id, integer in [0, 63)
	v_mbcnt_lo_u32_b32 v4, -1, 0
	v_mbcnt_hi_u32_b32 v4, -1, v4

        // v4 = v4 << 6, i.e. v4 *= 64
	v_lshlrev_b32_e32 v4, 6, v4

        // address is in v[0:1]
        // 64 bit addition of v4 to address
	v_add_co_u32_e32 v0, vcc, v0, v4
	v_addc_co_u32_e32 v1, vcc, 0, v1, vcc

        // Duplicate the 64 bit integer passed in v[2:3]
	v_mov_b32_e32 v4, v2
	v_mov_b32_e32 v5, v3

        // v[0:1] contains address for each lane to write to
        // v[2:5] contains (the same) 128 bit data to write
	flat_store_dwordx4 v[0:1], v[2:5]
	flat_store_dwordx4 v[0:1], v[2:5] offset:16
	flat_store_dwordx4 v[0:1], v[2:5] offset:32
	flat_store_dwordx4 v[0:1], v[2:5] offset:48

        // Wait for something

#endif
}

extern "C" __attribute__((visibility("default"))) int main(int argc,
                                                           char **argv)
{
  (void)argc;
  (void)argv;
  int differ = 0;

  uint64_t initial_value = 7;

  hostrpc::page_t page;

  // Initialize it as if by calling clear
  hostrpc::cacheline_t &line = page.cacheline[platform::get_lane_id()];
  for (unsigned e = 0; e < 8; e++)
    {
      line.element[e] = UINT64_MAX;
    }

  unsigned rep = 0;
  // for (unsigned rep = 0; rep < 64000; rep++)
  {
    {
      if (platform::get_lane_id() % 2 == 0)
        {
          for (unsigned e = 0; e < 8; e++)
            {
              line.element[e] =
                  initial_value + platform::get_lane_id() + e + rep;
              // expect.element[e] = 2 * (line.element[e] + 1);
            }

          hostcall_client(&line.element[0]);

          for (unsigned e = 0; e < 8; e++)
            {
              // differ += (line.element[e] != expect.element[e]);
            }
        }
    }
  }

  return differ;
}
#endif
