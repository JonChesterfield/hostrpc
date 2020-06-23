#include "base_types.hpp"
#include "catch.hpp"
#include "interface.hpp"

#include <cstring>
#include <thread>

static void init_page(hostrpc::page_t *page, uint64_t v)
{
  for (unsigned i = 0; i < 64; i++)
    {
      for (unsigned e = 0; e < 8; e++)
        {
          page->cacheline[i].element[e] = v;
        }
    }
}

TEST_CASE("x64_gcn_stress") { using namespace hostrpc; }
