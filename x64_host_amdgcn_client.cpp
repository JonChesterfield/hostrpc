#include "catch.hpp"
#include "hsa.hpp"

#include <string>

namespace
{
hsa::init global_state;
}

TEST_CASE("Is init")
{
  REQUIRE(global_state.status == HSA_STATUS_SUCCESS);

  std::vector<hsa_agent_t> kernel_agents;
  std::vector<hsa_agent_t> other_agents;
  hsa::iterate_agents([&](hsa_agent_t agent) -> hsa_status_t {
    auto features = hsa::agent_get_info_feature(agent);
    std::vector<hsa_agent_t>* list =
        (features & HSA_AGENT_FEATURE_KERNEL_DISPATCH) ? &kernel_agents
                                                       : &other_agents;
    list->push_back(agent);
    return HSA_STATUS_SUCCESS;
  });

  printf("Found %zu kernel agents\n", kernel_agents.size());
  printf("Found %zu other agents\n", other_agents.size());
}
