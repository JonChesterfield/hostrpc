#include "hsa.hpp"

#include <string>

int main_with_hsa()
{
  uint64_t agent_id = 0;
  hsa::iterate_agents([&](hsa_agent_t agent) -> hsa_status_t {
    auto name = hsa::agent_get_info_name(agent);
    std::string sname(name.begin(), name.size());

    printf("Agent %lu (%s)\n", agent_id++, sname.c_str());

    uint64_t region_id = 0;
    hsa::iterate_regions(agent, [&](hsa_region_t region) -> hsa_status_t {
      printf("  Region %lu (%zu KiB, %s, %s)\n", region_id++,
             hsa::region_get_info_size(region) / 1024,
             hsa::enum_as_str(hsa::region_get_info_segment(region)),
             hsa::enum_as_str(hsa::region_get_info_global_flags(region)));

      return HSA_STATUS_SUCCESS;
    });

    uint64_t pool_id = 0;

    hsa::iterate_memory_pools(agent,
                              [&](hsa_amd_memory_pool_t pool) -> hsa_status_t {
                                (void)pool;
                                printf("  Pool %lu\n", pool_id++);
                                return HSA_STATUS_SUCCESS;
                              });

    return HSA_STATUS_SUCCESS;
  });

  return 0;
}

int main()
{
  hsa::init hsa_state;
  return main_with_hsa();
}
